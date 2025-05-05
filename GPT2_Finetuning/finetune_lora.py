"""
Fine-tune GPT-2 (small) for cybersecurity summarisation **with LoRA adapters**.
Identical CLI to the original script:

  • `--search N` → Optuna HPO for N trials (minimising eval_loss)
  • `--retrain-best-model` → train once with the best params saved earlier

LoRA adds < 1% trainable parameters instead of ~124 M, so you get:
  • ~10× lower VRAM / RAM footprint
  • faster epochs on CPU / Apple M-series
  • the base GPT-2 weights stay frozen and reusable elsewhere
"""

# ────────────────────────────────────────────────────────────────────────────── #
# IMPORTS AND CONFIGURATION                                                      #
# Loads standard, ML, LoRA, and project-specific modules.                        #
# ────────────────────────────────────────────────────────────────────────────── #

from __future__ import annotations
import sys, argparse, json, shutil, gc, resource, multiprocessing as mp, os
from pathlib import Path

import optuna
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

# ── NEW: LoRA / PEFT ──────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model   #, PeftModel

# add project root for `config.py`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (   
    FT_GPT2_BASE_DIR,
    FT_GPT2_LORA_OUTPUT_ROOT,
    FT_GPT2_LORA_BEST_PARAMS_FILE,
    FT_GPT2_LORA_BEST_METRICS_FILE,
    FT_GPT2_LORA_ALL_TRIALS_FILE,
    FT_GPT2_LORA_BEST_MODEL_DIR,             
)

# ──────────────────────────────────────────────────────────────────────────────── #
# PARALLELISM, THREADING, AND RESOURCE LIMITS                                      #
# Configures threading, tokenizer parallelism, open file limits (macOS M1-safe).   #
# ──────────────────────────────────────────────────────────────────────────────── #

CPU_COUNT = mp.cpu_count()
torch.set_num_threads(CPU_COUNT)
torch.set_num_interop_threads(CPU_COUNT)
os.environ.update({
    "TOKENIZERS_PARALLELISM": "true",
    "OMP_NUM_THREADS": str(CPU_COUNT),
    "OPENBLAS_NUM_THREADS": str(CPU_COUNT),
    "VECLIB_MAXIMUM_THREADS": str(CPU_COUNT),
})
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft < 1024:
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))

# ──────────────────────────────────────────────────────────────────────────────── #
# CONSTANTS AND CONFIGURATION                                                      #
# Sets model name, paths, device, block size, number of workers, seed.             #
# ──────────────────────────────────────────────────────────────────────────────── #

MODEL_NAME = "gpt2"
BASE_DIR = FT_GPT2_BASE_DIR
DATA_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DATA_BASE_DIR / "DataPreprocessing"
BLOCK_SIZE = 512
OUTPUT_ROOT = FT_GPT2_LORA_OUTPUT_ROOT
BEST_PARAMS_FILE = FT_GPT2_LORA_BEST_PARAMS_FILE
BEST_METRICS_FILE = FT_GPT2_LORA_BEST_METRICS_FILE
ALL_TRIALS_FILE = FT_GPT2_LORA_ALL_TRIALS_FILE

DEVICE      = "mps" if torch.backends.mps.is_available() else (
              "cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
SEED        = 42
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────── #
# DATA LOADING AND TOKENIZATION HELPERS                                           #
# ─────────────────────────────────────────────────────────────────────────────── #

def load_splits() -> DatasetDict:
    splits = {}
    for split in ("train", "validation"):
        with (DATA_DIR / f"{split}.jsonl").open("r", encoding="utf-8") as fh:
            splits[split] = Dataset.from_list([json.loads(l) for l in fh])
    return DatasetDict(splits)

def _tokenise_single(ex, tok: GPT2Tokenizer):
    """
        Tokenizes a single example by creating a prompt + target string
        and masking prompt tokens with -100 in the labels.
    """
    tgt = ex["summary"]
    tgt_len = len(tok(tgt, add_special_tokens=False)["input_ids"])
    prompt = (f"<|summarize|> Document: {ex['source']}\n\n"
              f"### Task: Summarise the document for a cybersecurity analyst "
              f"in roughly the same number of {tgt_len} tokens.\n\n"
              f"### Summary: ")
    full = prompt + tgt
    enc  = tok(full, truncation=True, max_length=BLOCK_SIZE,
               padding="max_length")
    p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
    pad   = tok.pad_token_id
    enc["labels"] = [-100 if (i < p_len or t == pad) else t
                     for i, t in enumerate(enc["input_ids"])]
    return enc

def tokenise_batched(batch, tok: GPT2Tokenizer):
    """
    Tokenizes a batch of examples by applying _tokenise_single to each.

    Returns:
        Dictionary of tokenized fields with input_ids, attention_mask, labels.
    """
    encodings = [
        _tokenise_single({"source": s, "summary": t}, tok)
        for s, t in zip(batch["source"], batch["summary"])
    ]
    keys = encodings[0].keys()
    return {k: [e[k] for e in encodings] for k in keys}

# ──────────────────────────────────────────────────────────────────────────────── #
# TRAINER BUILDER WITH LoRA WRAPPING                                               #
#                                                                                  #
# Constructs a Hugging Face Trainer configured with LoRA adapters applied to GPT-2 #
# Combines pretrained GPT-2 base model with injected low-rank adapter layers.      #
# Returns Trainer ready for training, evaluation, or hyperparameter optimization.  #
# ──────────────────────────────────────────────────────────────────────────────── #

def build_trainer(trial_params: dict, tokenised: DatasetDict, out_dir: Path):
    """
    Constructs a Trainer configured for LoRA fine-tuning of GPT-2.

    Args:
        trial_params: hyperparameters (incl. tokenizer + LoRA params).
        tokenised: DatasetDict with tokenized train/validation splits.
        out_dir: output directory for checkpoints.

    Returns:
        Configured Hugging Face Trainer.
    """
    tokenizer = trial_params.pop("tokenizer")

    # ─── Define LoRA adapter configuration ───────────────────────────────── #
    lora_cfg = LoraConfig(
        r               = trial_params.pop("lora_r", 8),
        lora_alpha      = trial_params.pop("lora_alpha", 32),
        lora_dropout    = trial_params.pop("lora_dropout", 0.05),
        bias            = "none",
        target_modules  = ["c_attn", "c_proj"],     # GPT-2’s QKV projection layer
        task_type       = "CAUSAL_LM",
    )

    # ─── Load base model and apply LoRA adapters ─────────────────────────── #
    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model      = get_peft_model(base_model, lora_cfg).to(DEVICE)

    data_collator = DataCollatorWithPadding(tokenizer)
    
    training_args = TrainingArguments(
        output_dir           = str(out_dir),
        overwrite_output_dir = True,
        eval_strategy  = "epoch",
        save_strategy        = "epoch",
        logging_steps        = 100,
        save_total_limit     = 2,
        dataloader_num_workers = NUM_WORKERS,
        report_to            = [],
        fp16                 = torch.cuda.is_available(),
        seed                 = SEED,
        load_best_model_at_end = False,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,
        remove_unused_columns = False,
        **trial_params
    )

    return Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenised["train"],
        eval_dataset    = tokenised["validation"],
        data_collator   = data_collator,
        tokenizer       = tokenizer,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

# ─────────────────────────────────────────────────────────────────────────────── #
# OPTUNA OBJECTIVE FUNCTION                                                       #
#                                                                                 #
# Defines a single hyperparameter optimization trial for LoRA fine-tuning.        #
# Each trial samples a configuration, trains the model, evaluates it,             #
# and returns validation loss as the optimization metric.                         #
# ─────────────────────────────────────────────────────────────────────────────── #

def objective(trial: optuna.Trial, tokenised: DatasetDict, tok: GPT2Tokenizer):
    """
    Optuna objective function for LoRA hyperparameter tuning.

    Each trial samples a configuration of both standard training parameters
    and LoRA adapter parameters, fine-tunes the model, evaluates validation loss,
    and returns eval_loss as the optimization metric.

    Args:
        trial: Optuna Trial object to sample hyperparameters.
        tokenised: Tokenized dataset (train + validation splits).
        tok: GPT-2 tokenizer instance.

    Returns:
        Validation loss (float) to be minimized by Optuna.
    """

    # ─── Sample standard hyperparameters ───────────────────────────────── #
    lr   = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    wd   = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
    ep   = trial.suggest_int("num_train_epochs", 1, 5)
    bsz  = trial.suggest_categorical("per_device_train_batch_size", [4])
    gacc = trial.suggest_categorical("gradient_accumulation_steps", [2])

    # ─── Sample LoRA-specific hyperparameters ─────────────────────────── #
    r = trial.suggest_categorical("lora_r", [8, 16, 32])
    alp = trial.suggest_categorical("lora_alpha", [8, 16, 32, 48, 64, 96, 128])     


    run_dir = OUTPUT_ROOT / f"trial_{trial.number}"

    # ─── Build Trainer with sampled hyperparameters ─────────────────────# 
    trainer = build_trainer(
        dict(tokenizer=tok,
             learning_rate=lr,
             weight_decay=wd,
             num_train_epochs=ep,
             per_device_train_batch_size=bsz,
             gradient_accumulation_steps=gacc,
             lora_r=r, 
             lora_alpha=alp
        ),
        tokenised, run_dir,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trial.set_user_attr("metrics", metrics)
    val_loss = metrics["eval_loss"]

    # ─── Explicit cleanup to free GPU/CPU memory before next trial ───── #
    trainer.model.cpu(); del trainer # move the model to CUP; delete Trainer object
    torch.cuda.empty_cache(); gc.collect()

    return val_loss

# ────────────────────────────────────────────────────────────────────────────────--------- #
# MAIN ENTRY POINT                                                                          #
#                                                                                           #
# Orchestrates CLI-based workflow: handles retrain, Optuna search, or single-run mode.      #
# Responsible for tokenizer loading, dataset preprocessing, directory setup, and dispatch.  #
# Command-line interface mirrors classic finetuning script for consistency.                 #
# ────────────────────────────────────────────────────────────────────────────────--------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=int, default=0,
                    help="Run Optuna with N trials (0 = no search).")
    ap.add_argument("--retrain-best-model", action="store_true",
                    help="Retrain with the stored best hyper-params.")
    args = ap.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_splits()
    tokenised = raw.map(lambda ex: tokenise_batched(ex, tokenizer),
                        batched=True, batch_size=16, remove_columns=["source", "summary"])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ─── Only retrain the recorded best trial ────────────────────────────
    if args.retrain_best_model:
        with open(BEST_PARAMS_FILE) as fh:
            best = json.load(fh)
        best["tokenizer"] = tokenizer
        out_dir = FT_GPT2_LORA_BEST_MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer = build_trainer(best, tokenised, out_dir)
        trainer.train(); trainer.save_model(out_dir)
        trainer.tokenizer.save_pretrained(out_dir)
        json.dump(trainer.evaluate(), open(BEST_METRICS_FILE, "w"), indent=2)
        print(f"[DONE] model → {out_dir}")
        return

    # ─── Optuna search ───────────────────────────────────────────────────
    if args.search > 0:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: objective(t, tokenised, tokenizer),
                       n_trials=args.search)
        best = study.best_trial
        BEST_PARAMS_FILE.write_text(json.dumps(best.params, indent=2))
        BEST_METRICS_FILE.write_text(json.dumps(best.user_attrs["metrics"], indent=2))
        with open(ALL_TRIALS_FILE, "w") as fh:
            for t in study.trials:
                fh.write(json.dumps({
                    "number": t.number, "value": t.value, "params": t.params, "metrics": t.user_attrs["metrics"],
                }))
        print(f"Best eval_loss {best.value:.4f} → params saved to {BEST_PARAMS_FILE}")
        return

    # ─── Single one-off run (no HPO) ─────────────────────────────────────
    default = dict(
        tokenizer=tokenizer,
        learning_rate=5e-4,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
    )
    trainer = build_trainer(default, tokenised, OUTPUT_ROOT / "single_run")
    trainer.train(); print("Metrics:", trainer.evaluate())
    trainer.save_model(OUTPUT_ROOT / "single_run")
    tokenizer.save_pretrained(OUTPUT_ROOT / "single_run")

if __name__ == "__main__":
    main()
