import argparse, time, json, csv, os, gc, sys, subprocess
from typing import List, Dict, Any
from torch.amp import GradScaler, autocast
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
)

import yaml

_BACKEND_MAP = {
    "default": {},
    "flash": {"attn_implementation": "flash_attention_2"},
    "sdpa": {"attn_implementation": "sdpa"},
    "xformers": {"attn_implementation": "xformers"}
}

class JsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 768, padding: str = "longest"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.samples = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                text = obj["source"] + tokenizer.eos_token + obj["summary"]
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.samples[idx],
            truncation=True,
            padding=self.padding,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )

def get_loader(path: str, tokenizer, batch_size: int, cfg: Dict[str, Any]):
    padding = cfg.get("padding", "longest")  # ← default to dynamic padding
    max_length = int(cfg.get("max_length", 768)) # ← default is max_length 768; small GPT-2
    ds = JsonlDataset(path, tokenizer, padding=padding)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True
    )

def load_model(cfg: Dict[str, Any], device):
    extra = _BACKEND_MAP.get(cfg.get("attn_backend", "default"), {})
    model_name = cfg.get("model_name", "gpt2")

    wants_custom_arch = any(k in cfg for k in ("layers", "heads"))
    if wants_custom_arch:
        base_cfg = AutoConfig.from_pretrained(model_name)
        if "layers" in cfg:
            base_cfg.n_layer = int(cfg["layers"])
        if "heads" in cfg:
            base_cfg.n_head = int(cfg["heads"])
        model = AutoModelForCausalLM.from_config(base_cfg, **extra)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **extra)

    precision = cfg.get("precision", "fp32")

    compile_time = 0.0
    if cfg.get("compile"):
        mode = cfg.get("compile_mode", "default")
        start = time.perf_counter()
        model = torch.compile(model, mode=mode)
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - start

    return model.to(device), compile_time

def train_benchmark(model, loader, cfg, device, steps=500):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = cfg.get("precision") == "fp16"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    token_total, times, peak_mem = 0, [], 0
    model.train()
    data_iter = iter(loader)

    gpu_utils = []
    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update(); opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        token_total += batch["input_ids"].numel()
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))

        try:
            smi_out = subprocess.check_output([
                "nvidia-smi", "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"
            ])
            util = int(smi_out.decode().splitlines()[0])
            gpu_utils.append(util)
        except Exception:
            pass

    return {
        "steps": steps,
        "avg_step_time": sum(times) / len(times),
        "tokens_per_sec": token_total / sum(times),
        "max_vram": peak_mem / 1e6,
        "gpu_utilization_avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else -1
    }

def load_experiment_matrix(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yml", ".yaml"):
        if yaml is None:
            sys.exit("pyyaml not installed – `pip install pyyaml` or use JSON")
        with open(path, "r") as fh:
            return yaml.safe_load(fh)
    elif ext == ".json":
        with open(path, "r") as fh:
            return json.load(fh)
    else:
        sys.exit("Unsupported config file format; use .yaml, .yml or .json")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--csv", default="results.csv")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--configs")
    args = p.parse_args()

    exp_matrix = load_experiment_matrix(args.configs) if args.configs else DEFAULT_CONFIGS
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {}
    fieldnames = [
        "name", 
        "model_name", 
        "precision", 
        "compile", 
        "attn_backend", 
        "batch_size",
        "steps", 
        "avg_step_time", 
        "tokens_per_sec", 
        "max_vram",
        "compile_warmup_time", 
        "gpu_utilization_avg", 
        "padding", 
        "max_length"
    ]

    new_file = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for cfg in exp_matrix:
            bs = int(cfg.get("batch_size", 4))
            if bs not in loaders:
                loaders[bs] = get_loader(args.dataset, tokenizer, bs, cfg)
            loader = loaders[bs]

            torch.cuda.empty_cache(); gc.collect()
            model, compile_time = load_model(cfg, device)

            stats = train_benchmark(model, loader, cfg, device, steps=args.steps)
            row = {k: cfg.get(k) for k in (
                "name", "model_name", "precision", "compile", "attn_backend", "batch_size", "padding", "max_length")}
            row.update(stats)
            row["compile_warmup_time"] = compile_time
            writer.writerow(row)
            fh.flush()
            print(row)

            del model; torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()

