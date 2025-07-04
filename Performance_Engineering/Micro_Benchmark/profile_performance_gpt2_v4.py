#!/usr/bin/env python3
"""
GPT-2 micro-benchmark with detailed timing of:
  - offline tokenisation
  - dataloader wait
  - collator latency
  - transfer, forward, backward, optimizer, sync
  - peak GPU and CPU memory
  - optional nvidia-smi overhead tracking

Requires: torch, transformers, pyyaml, psutil

run: python3 profiling_performance_gpt2_v4.py \
    --dataset ../DatasetPreprocessing/train.jsonl \
    --configs ./config.yaml \
    --steps 500 \
    --csv results.csv

monitor (tensorboard): tensorboard --logdir=./logdir --bind_all
"""
import argparse, time, json, csv, os, gc, sys, subprocess
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import trange
from torch.amp import GradScaler, autocast
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
)

import psutil
import yaml

from torch.profiler import record_function

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

        with open(path, "r", encoding="utf-8") as fh:
            raw_texts = [
                json.loads(line)["source"] + tokenizer.eos_token + json.loads(line)["summary"]
                for line in fh
            ]

        t0 = time.perf_counter()
        self.samples = [
            tokenizer(t, truncation=True, padding=padding, max_length=max_length, return_special_tokens_mask=True)
            for t in raw_texts
        ]
        self.tokenization_time = time.perf_counter() - t0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_loader(path: str, tokenizer, batch_size: int, cfg: Dict[str, Any]):
    padding = cfg.get("padding", "longest")
    max_length = int(cfg.get("max_length", 768))
    ds = JsonlDataset(path, tokenizer, max_length=max_length, padding=padding)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator_stats = {"time": 0.0}

    def timed_collate(batch):
        t0 = time.perf_counter()
        result = collator(batch)
        collator_stats["time"] += time.perf_counter() - t0
        return result

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=timed_collate,
                      pin_memory=torch.cuda.is_available(), num_workers=0), ds, collator_stats

def load_model(cfg: Dict[str, Any], device):
    extra = _BACKEND_MAP.get(cfg.get("attn_backend", "default"), {})
    model_name = cfg.get("model_name", "gpt2")

    if any(k in cfg for k in ("layers", "heads")):
        base_cfg = AutoConfig.from_pretrained(model_name)
        if "layers" in cfg:
            base_cfg.n_layer = int(cfg["layers"])
        if "heads" in cfg:
            base_cfg.n_head = int(cfg["heads"])
        model = AutoModelForCausalLM.from_config(base_cfg)
        model.config.update(extra)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **extra)

    compile_time = 0.0
    if cfg.get("compile") and device.type == "cuda":
        mode = cfg.get("compile_mode", "default")
        t0 = time.perf_counter()
        model = torch.compile(model, mode=mode)
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t0

    return model.to(device), compile_time

def train_benchmark(model, loader, dataset, collator_stats, cfg, device, steps=500):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = device.type == "cuda" and cfg.get("precision") == "fp16"
    scaler = GradScaler(enabled=use_amp)

    token_total = 0
    step_times = []
    peak_gpu_mem = 0
    peak_cpu_mem = 0
    gpu_utils = []

    # -------- Manual Timing to complement tensorboard logging ------ #
    dataloader_time = 0.0
    transfer_time   = 0.0
    forward_time    = 0.0
    backward_time   = 0.0
    optimizer_time  = 0.0
    sync_time       = 0.0
    # --------------------------------------------------------------- #

    proc = psutil.Process() if psutil else None

    model.train()
    data_iter = iter(loader)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logdir", run_name)
    
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ]
    )
    profiler.start()

    for step in trange(steps, desc="Benchmark Steps", dynamic_ncols=True):
        profiler.step()

        # --------- Dataloader ------------ #
        t0 = time.perf_counter()
        with torch.profiler.record_function("dataloader"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
        dataloader_time += time.perf_counter() - t0

        # ---------- Transfer ------------ #
        t1 = time.perf_counter()
        with torch.profiler.record_function("transfer"):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
        transfer_time += time.perf_counter() - t1

        # --------- Forward -------------- #
        t2 = time.perf_counter()
        with torch.profiler.record_function("forward"):
            with autocast(device_type='cuda', enabled=use_amp):
                loss = model(**batch).loss
        forward_time += time.perf_counter() - t2

        # --------- Backward ------------- #
        t3 = time.perf_counter()
        with torch.profiler.record_function("backward"):
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        backward_time += time.perf_counter() - t3

        # -------- Optimizer ------------- #
        t4 = time.perf_counter()
        with torch.profiler.record_function("optimizer"):
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        optimizer_time += time.perf_counter() - t4

        # ---------- Sync --------------- #
        t5 = time.perf_counter()
        with torch.profiler.record_function("sync"):
            if device.type == "cuda":
                torch.cuda.synchronize()
        sync_time += time.perf_counter() - t5

        token_total += batch["input_ids"].numel()
        if device.type == "cuda":
            peak_gpu_mem = max(peak_gpu_mem, torch.cuda.max_memory_allocated(device))
        if proc:
            peak_cpu_mem = max(peak_cpu_mem, proc.memory_info().rss)

        if device.type == "cuda" and step % 10 == 0:
            try:
                util = int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    timeout=1
                ).decode().strip())
                gpu_utils.append(util)
            except Exception:
                pass

    profiler.stop()

    print(f"\n── Profiling summary ({steps} steps) ──")
    print(f"Offline tokenisation time : {dataset.tokenization_time:8.4f} s")
    print(f"Collator CPU time         : {collator_stats['time']:8.4f} s")
    print(f"DataLoader queue wait     : {dataloader_time:8.4f} s")
    print(f"Transfer to device        : {transfer_time:8.4f} s")
    print(f"Forward pass              : {forward_time:8.4f} s")
    print(f"Backward pass             : {backward_time:8.4f} s")
    print(f"Optimiser step            : {optimizer_time:8.4f} s")
    print(f"CUDA sync                 : {sync_time:8.4f} s")
    print(f"Peak GPU VRAM             : {peak_gpu_mem / 1e6:8.1f} MB")
    print(f"Peak CPU RAM              : {peak_cpu_mem / 1e6:8.1f} MB" if proc else "CPU RAM peak             : n/a")

    return {
        "steps": steps,
        "avg_step_time": -1,
        "tokens_per_sec": token_total / steps,
        "max_vram": peak_gpu_mem / 1e6,
        "max_cpu_ram": peak_cpu_mem / 1e6 if proc else -1,
        "gpu_utilization_avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else -1,
        "collator_time": collator_stats["time"]
    }

def load_experiment_matrix(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        if yaml is None:
            sys.exit("pyyaml not installed – install it or use JSON instead.")
        with open(path, "r") as fh:
            return yaml.safe_load(fh)
    elif ext == ".json":
        with open(path, "r") as fh:
            return json.load(fh)
    else:
        sys.exit("Unsupported config file format")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--configs")
    args = ap.parse_args()

    exp_matrix = load_experiment_matrix(args.configs) if args.configs else [{}]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {}
    fieldnames = sorted(set().union(*[cfg.keys() for cfg in exp_matrix]) | {
        "steps", "avg_step_time", "tokens_per_sec", "max_vram", "max_cpu_ram",
        "compile_warmup_time", "gpu_utilization_avg", "collator_time"
    })

    new_file = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for cfg in exp_matrix:
            bs = int(cfg.get("batch_size", 4))
            padding = cfg.get("padding", "longest")
            max_len = int(cfg.get("max_length", 768))
            key = (bs, padding, max_len)

            if key not in loaders:
                loaders[key] = get_loader(args.dataset, tokenizer, bs, cfg)
            loader, dataset, collator_stats = loaders[key]

            torch.cuda.empty_cache(); gc.collect()
            try:
                model, compile_time = load_model(cfg, device)
                stats = train_benchmark(model, loader, dataset, collator_stats, cfg, device, steps=args.steps)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on config: {cfg.get('name', 'unknown')}")
                continue

            row = {**cfg, **stats, "compile_warmup_time": compile_time}
            writer.writerow(row); fh.flush()
            print("✓", row.get("name", key))
            print(json.dumps(row, indent=2))

            del model; torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
