#!/usr/bin/env python3
"""
Fine-tune a custom GPT-2 checkpoint for cybersecurity summarisation.

Example
-------
python finetuned_custom_gpt2.py \
    --checkpoint GroundUp_ModelTraining_Outcome/gpt2_wikitext_256.pt \
    --config     configs/fine_tune_custom_gpt2.yaml \
    --prep       data/prepare_finetuning_custom_gpt2.py
"""
import argparse, importlib.util, resource
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup

from config_utils import load_config
from model import GPT


# ────────────────────────────────────────────────────────────────────────
def _set_rlimit(min_soft=2048):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < min_soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(min_soft, hard), hard))
        print(f"open-file limit ↑ to {min(min_soft, hard)}")


def _dyn_import(path):
    """Dynamically import a prepare_dataset function from `path`."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # type: ignore
    return mod.load_and_prepare_dataset


# ──────────────────────────────────────────────────────────────────────────────── #
# MAIN FUNCTION                                                                    #
#                                                                                  #
# Orchestrates fine-tuning of a custom GPT-2 checkpoint on prepared dataset.       #
# Loads config, dataset, checkpoint, optimizer, scheduler, and runs training loop. #
# ──────────────────────────────────────────────────────────────────────────────── #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--prep",       required=True)
    args = p.parse_args()

    _set_rlimit()

    cfg = load_config(args.config)
    tok = GPT2Tokenizer.from_pretrained(cfg["tokenizer"]["name"])
    tok.pad_token = tok.eos_token

    (train_ids, train_lbls), (val_ids, val_lbls) = _dyn_import(args.prep)(cfg)

    model_cfg = {"vocab_size": tok.vocab_size, **cfg["model"]}
    model = GPT(model_cfg)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    # ── DataLoaders ────────────────────────────────────────────────────
    def _mk_loader(x, y, shuffle):
        ds = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=shuffle,
            num_workers=0,                # keep 0 workers on macOS
            pin_memory=False,
        )

    train_loader = _mk_loader(train_ids, train_lbls, shuffle=True)
    val_loader   = _mk_loader(val_ids,   val_lbls, shuffle=False)

    # ── optimiser & scheduler ──────────────────────────────────────────
    lr = float(cfg["train"]["learning_rate"])
    wd = float(cfg["train"].get("weight_decay", 0.01))

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = len(train_loader) * cfg["train"]["num_epochs"]
    sched = get_cosine_schedule_with_warmup(
        optim,
        int(cfg["train"].get("warmup_ratio", 0.06) * total_steps),
        total_steps,
    )
    crit = torch.nn.CrossEntropyLoss(ignore_index=-100)

    best, no_up = float("inf"), 0
    patience = cfg["train"].get("early_stopping_patience", 3)

    # ── training loop ──────────────────────────────────────────────────
    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        model.train(); run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # --- shifted-loss to avoid off-by-one -------------------- #
            logits = model(x)
            shift_logits = logits[:, :-1, :].contiguous()          ### fixed
            shift_labels = y[:, 1:].contiguous()                   ### fixed
            loss = crit(
                shift_logits.view(-1, shift_logits.size(-1)),      ### fixed
                shift_labels.view(-1),                             ### fixed
            )
            # --------------------------------------------------------- #

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            run += loss.item()
        tr_loss = run / len(train_loader)

        # ---- val ----
        model.eval(); run = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                shift_logits = logits[:, :-1, :].contiguous()   
                shift_labels = y[:, 1:].contiguous()              
                loss = crit(
                    shift_logits.view(-1, shift_logits.size(-1)),  
                    shift_labels.view(-1),                         
                )
                run += loss.item()
        val_loss = run / len(val_loader)
        print(f"Epoch {epoch}: train={tr_loss:.4f} | val={val_loss:.4f}")

        # ---- early stopping ----
        if val_loss < best - 1e-4:
            best, no_up = val_loss, 0
        else:
            no_up += 1
            if no_up >= patience:
                print("Early stopping."); break

    # ─── Save fine-tuned model ─────────────────────────────────────────────── #
    out_dir = Path(cfg["train"].get("model_output_path",
                                    "GroundUp_ModelTraining_Outcome"))
    out_dir.mkdir(parents=True, exist_ok=True)                  
    save_path = out_dir / f"finetuned_custom_gpt2_summariser.pt"
    torch.save(model.state_dict(), save_path)                      
    print(f"Model saved → {save_path}")


if __name__ == "__main__":
    main()

