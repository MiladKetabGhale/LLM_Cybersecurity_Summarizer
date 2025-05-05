#!/usr/bin/env python3
"""Fine‑tune GPT‑2 for cybersecurity summarisation **with optional Optuna HPO**.

• Original single‑run behaviour remains the default.
• Pass `--search N` to run an Optuna study with **N trials**.
• **No automatic retrain** after the search; the script simply prints and
  saves the best hyper‑parameters *and* their evaluation metrics.

Example
-------
$ python unified_finetune.py --search 20
"""

# ───────────────────────────────────────────────────────────────────────────── #
# IMPORTS AND CONFIGURATION                                                     #
# This section loads libraries, paths, constants, and parallelism settings.     #
# ───────────────────────────────────────────────────────────────────────────── #

from __future__ import annotations
import sys
import argparse
import json, shutil
from pathlib import Path

import optuna
import torch
import os, multiprocessing as mp, torch, gc, resource
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FT_GPT2_BASE_DIR, 
    FT_GPT2_OUTPUT_ROOT, 
    FT_GPT2_BEST_PARAMS_FILE, 
    FT_GPT2_BEST_METRICS_FILE,
    FT_GPT2_BEST_MODEL_DIR,
    FT_GPT2_ALL_TRIALS_FILE
)

# ────────────────────────────────────────────────────────────────────────────────-- #
# MULTI-THREADING, RESOURCE LIMITS, AND ENVIRONMENT CONFIGURATION                    #
# Sets up maximum CPU/thread parallelism, tokenizer parallelism, and system limits.  #
# Optimized for macOS M1/Pro.                                                        #
# ────────────────────────────────────────────────────────────────────────────────-- #

# How many logical CPUs do we have?
CPU_COUNT = mp.cpu_count()                 # e.g. 8‑core M1 Pro → 8

# Torch parallelism
torch.set_num_threads(CPU_COUNT)           # intra‑op (matrix ops)
torch.set_num_interop_threads(CPU_COUNT)   # inter‑op (operator fusion)

# Hugging‑Face tokenizer threads
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# OpenBLAS & Accelerate (Apple BLAS) threads for fallback numpy ops
os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_COUNT)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_COUNT)  # Apple vecLib / Accelerate

# ─── FD & multiprocessing safety patches ────────────────────────────────
try:
    mp.set_start_method("spawn", force=True)  # safer on macOS for DataLoader
except RuntimeError:
    # start method already set (e.g. interactive notebook)
    pass

# Raise soft open‑file limit to 1024 or higher (up to 4048) if macOS default (256) is too low
soft_no_file, hard_no_file = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft_no_file < 1024:
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard_no_file))


# ──────────────────────────────────────────────────────────────────────────────── #
# CORE CONFIGURATION CONSTANTS                                                     #
# Sets model names, directories, device selection, and training constants.         #
# ──────────────────────────────────────────────────────────────────────────────── #

MODEL_NAME = "gpt2"
BASE_DIR = FT_GPT2_BASE_DIR
DATA_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DATA_BASE_DIR / "DataPreprocessing"
BLOCK_SIZE = 512
OUTPUT_ROOT = FT_GPT2_OUTPUT_ROOT
BEST_PARAMS_FILE = FT_GPT2_BEST_PARAMS_FILE
BEST_METRICS_FILE = FT_GPT2_BEST_METRICS_FILE
ALL_TRIALS_FILE = FT_GPT2_ALL_TRIALS_FILE

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_WORKERS = 2
SEED = 42
torch.manual_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────── #
# DATA LOADING FUNCTIONS                                                           #
# Loads the train and validation splits from JSONL files and tokenizes them.       #
# ──────────────────────────────────────────────────────────────────────────────── #

def load_splits() -> DatasetDict:
    """Reads train/validation JSONL into a `DatasetDict`."""
    splits: dict[str, Dataset] = {}
    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as fh:
            splits[split] = Dataset.from_list([json.loads(line) for line in fh])
    return DatasetDict(splits)

def tokenize_fn(example: dict, tokenizer: GPT2Tokenizer):
    """Build *prompt + target* and mask prompt tokens in `labels`."""
    target = example["summary"]
    target_len = len(tokenizer(target, add_special_tokens=False)["input_ids"])

    prompt = (
        f"<|summarize|> Document: {example['source']}\n\n"
        f"### Task: Summarise the document for a cybersecurity analyst in roughly same {target_len} tokens.\n\n"
        f"### Summary: "
    )
    full = prompt + target

    enc = tokenizer(
        full,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
        return_tensors=None,
    )

    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels = [
        -100 if (i < prompt_len or tok == tokenizer.pad_token_id) else tok
        for i, tok in enumerate(enc["input_ids"])
    ]
    enc["labels"] = labels
    return enc


def tokenize_batched(batch, tokenizer: GPT2Tokenizer):
    """
    Tokenize a batch of examples into input_ids and masked labels for GPT-2 fine-tuning.

    For each example:
    - Build a prompt + target string.
    - Tokenize the combined text.
    - Mask prompt tokens and padding tokens in the label array with -100.

    Args:
        batch: Dictionary of lists with keys 'source' and 'summary' from Dataset.
        tokenizer: Hugging Face GPT2Tokenizer instance.

    Returns:
        Dictionary with tokenized input_ids, attention_mask, and masked labels.
    """

    sources  = batch["source"]
    targets  = batch["summary"]

    full_texts  = []
    prompt_lens = []

    # ─── Build prompt + target strings for each example ─────────────────── #
    for src, tgt in zip(sources, targets):
        tgt_len = len(tokenizer(tgt, add_special_tokens=False)["input_ids"])
        prompt  = (
            f"<|summarize|> Document: {src}\n\n"
            f"### Task: Summarise the document for a cybersecurity analyst "
            f"in roughly one third of {tgt_len} tokens.\n\n"
            f"### Summary: "
        )
        full_texts.append(prompt + tgt)
        prompt_lens.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))

    # ─── Tokenize the entire batch of prompt+target strings at once ─────── #
    enc = tokenizer(
        full_texts,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
    )

    # Build label masks: -100 for prompt tokens + padding, ids for summary
    labels = []
    pad_id = tokenizer.pad_token_id
    for ids, p_len in zip(enc["input_ids"], prompt_lens):
        labels.append(
            [-100 if (i < p_len or tok == pad_id) else tok
             for i, tok in enumerate(ids)]
        )

    enc["labels"] = labels
    return enc

# ──────────────────────────────────────────────────────────────────────────────── #
# COLLATOR CLASS FOR FIXED-LENGTH BATCHES                                          #
# Pads input_ids to BLOCK_SIZE and applies -100 mask for ignored label positions.  #
# ──────────────────────────────────────────────────────────────────────────────── #

class MaskPadCollator:
    """
    • Assumes *input_ids* are already right‑padded to BLOCK_SIZE.
    • Pads the *labels* field to the same length and converts padding
      tokens to ‑100 so the language‑model loss ignores them.
    """
    def __init__(self, tokenizer):
        self.base   = DataCollatorWithPadding(tokenizer)   # stacks tensors
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        # Pull labels out so the base collator won't modify them
        labels = [f.pop("labels") for f in features]        # list[list[int]]

        # Collate everything else (input_ids, attention_mask, …)
        batch  = self.base(features)                       # dict[str, tensor]

        # Pad labels to BLOCK_SIZE and apply ‑100 mask
        max_len = batch["input_ids"].shape[1]              # == BLOCK_SIZE
        padded  = [
            lbl + [self.pad_id] * (max_len - len(lbl))
            for lbl in labels
        ]
        labels_tensor = torch.tensor(padded)
        labels_tensor = labels_tensor.masked_fill(
            labels_tensor == self.pad_id, -100
        )
        batch["labels"] = labels_tensor
        return batch

# ──────────────────────────────────────────────────────────────────────────────── #
# TRAINER BUILDER                                                                  #
# Builds a Hugging Face Trainer configured with the given hyperparameters,         #
# dataset splits, model, tokenizer, data collator, and early stopping.             #
# ──────────────────────────────────────────────────────────────────────────────── #

def build_trainer(
    """
    Constructs a Hugging Face Trainer instance for fine-tuning GPT-2.

    Args:
        trial_params: Dictionary of hyperparameters for this training run.
                      (must include 'tokenizer' object inside; it will be popped)
        tokenized: Tokenized DatasetDict with 'train' and 'validation' splits.
        output_dir: Directory path where checkpoints and outputs will be saved.

    Returns:
        A configured Hugging Face Trainer instance.
    """
    trial_params: dict,
    tokenized: DatasetDict,
    output_dir: Path,
):
    tokenizer = trial_params.pop("tokenizer")  # required
   
    # ─── Create custom collator to pad input_ids and mask prompt tokens in labels ─- #
    data_collator = MaskPadCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),         # where to save checkpoints
        overwrite_output_dir=True,          # overwrite any existing directory
        eval_strategy="epoch",              # evaluate every epoch
        save_strategy="epoch",
        logging_steps=100,                  # log metrics every 100 steps
        save_total_limit=2,                 # keep only last 2 checkpoints
        dataloader_num_workers=NUM_WORKERS, # parallel DataLoader workers
        report_to=[],                       # disable external loggers (add MLflow/W&B if desired)
        fp16=torch.cuda.is_available(),
        seed=SEED,
        # ── needed for EarlyStoppingCallback ───────────────────────────
        load_best_model_at_end=True,     # reload checkpoint with lowest loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # ─── Add trial-specific hyperparameters (learning_rate, etc.) ──────── #
        **trial_params,
    )

    # ─── Load pretrained GPT-2 model and move to target device (CPU/GPU/MPS) ── #
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # ─── Return fully configured Trainer with datasets, collator, callbacks ─── #
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

# ────────────────────────────────────────────────────────────────────────────────-- #
# OPTUNA OBJECTIVE FUNCTION                                                          #
# Defines the logic for a single trial of hyperparameter optimization.               #
# For each trial: sample hyperparameters → train model → evaluate → return eval_loss #
# ────────────────────────────────────────────────────────────────────────────────-- #

def objective(trial: optuna.Trial, tokenized: DatasetDict, tokenizer: GPT2Tokenizer):
    """
    Objective function for Optuna hyperparameter tuning.

    Each trial samples a combination of hyperparameters, builds a Trainer,
    trains the model, evaluates it on validation set, and returns eval_loss.

    Args:
        trial: An Optuna Trial object for hyperparameter suggestions.
        tokenized: Hugging Face DatasetDict containing tokenized train/validation sets.
        tokenizer: Hugging Face tokenizer instance used for decoding and tokenization.

    Returns:
        Evaluation loss (float) on validation set, used as optimization target.
    """

    # ─── Sample hyperparameters for this trial ──────────────────────────────── #
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
    epochs = trial.suggest_int("num_train_epochs", 1, 5)
    batch = trial.suggest_categorical("per_device_train_batch_size", [2,4])
    grad_acc = trial.suggest_categorical("gradient_accumulation_steps", [2,4])

    # ─── Define output directory for this trial's checkpoints ───────────────── #
    run_dir = OUTPUT_ROOT / f"trial_{trial.number}"

    # ─── Build a new Trainer instance with sampled hyperparameters ──────────── #
    trainer = build_trainer(
        dict(
            tokenizer=tokenizer,
            learning_rate=lr,
            weight_decay=wd,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=grad_acc,
        ),
        tokenized,
        run_dir,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # ─── store metrics so we can retrieve them later ───────────────────────
    trial.set_user_attr("metrics", metrics)
    
    # ─── Explicit cleanup to release GPU/CPU memory before next trial ────────
    val_loss = metrics["eval_loss"]
    trainer.model.cpu()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # invoke garbage collection to free memory
    gc.collect()
    
    # we keep eval_loss as the optimisation target
    return val_loss


# ───────────────────────────────────────────────────────────────────────────── #
# MAIN ENTRY POINT                                                              #
# Handles command-line arguments and controls execution flow.                   #
# ───────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search",
        type=int,
        default=0,
        help="Run Optuna HPO for N trials (0 = regular fine‑tune)",
    )

    # ─────– optional one‑shot retrain of the best trial ────────────
    parser.add_argument(
        "--retrain-best-model",
        action="store_true",
        help="Skip search / default run and retrain with the best params "
             "stored in config.FT_GPT2_BEST_PARAMS_FILE.",
    )
    # ────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    # --------------------------------------------------------------------
    # If you only want to retrain the recorded best trial, do that
    # and exit.  All other behaviour (search / default run) is untouched.
    # --------------------------------------------------------------------
    if args.retrain_best_model:

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        raw = load_splits()
        tokenized = raw.map(
            lambda ex: tokenize_batched(ex, tokenizer),
            batched=True,
            batch_size=16,
        )

        # ── pull hyper‑params from the JSON file we persisted earlier ──
        with open(FT_GPT2_BEST_PARAMS_FILE, "r", encoding="utf-8") as fh:
            best_params = json.load(fh)

        best_params["tokenizer"] = tokenizer      # build_trainer expects this

        out_dir = FT_GPT2_BEST_MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Re‑training best model → {out_dir}")
        print(f"\n Hyper‑parameters → {best_params}")

        trainer = build_trainer(best_params, tokenized, out_dir)
        trainer.train()

        # save model and metrics in the standard places
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        metrics = trainer.evaluate()
        with open(FT_GPT2_BEST_METRICS_FILE, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        print(f"[DONE] Model saved to {out_dir}")
        print(f"\n Metrics → {config.FT_GPT2_BEST_METRICS_FILE}")
        return                                    # finished

    # --------------------------------------------------------------------
    # Existing workflow starts here (unchanged).
    # --------------------------------------------------------------------
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_splits()
    tokenized = raw.map(lambda ex: tokenize_batched(ex, tokenizer), batched=True, batch_size=16)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # ── Hyper‑parameter search ────────────────────────────────────────────
    if args.search > 0:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(
            lambda t: objective(t, tokenized, tokenizer),
            n_trials=args.search,
        )

        best = study.best_trial
        best_metrics = best.user_attrs.get("metrics", {})
        print(
            f"Best trial #{best.number} – eval_loss={best.value:.4f}\n"
            f"Params: {best.params}\nMetrics: {best_metrics}"
        )

        # ── persist best params & metrics ────────────────────────────────
        BEST_PARAMS_FILE.write_text(json.dumps(best.params, indent=2))
        BEST_METRICS_FILE.write_text(json.dumps(best_metrics, indent=2))
        print(f"Best params → {BEST_PARAMS_FILE}")
        print(f"Best metrics → {BEST_METRICS_FILE}")

        # ── persist every trial’s summary for later analysis ─────────────
        with ALL_TRIALS_FILE.open("w", encoding="utf-8") as fh:
            for t in study.trials:
                entry = {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "metrics": t.user_attrs.get("metrics", {}),
                }
                fh.write(json.dumps(entry) + "\n")
        print(f"All trials summary → {ALL_TRIALS_FILE}")

    # ── Single‑run default (no search) ───────────────────────────────────
    else:
        default_params = dict(
            tokenizer=tokenizer,
            learning_rate=5e-4,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            weight_decay=0.0,
        )
        trainer = build_trainer(default_params, tokenized, OUTPUT_ROOT / "single_run")
        trainer.train()
        metrics = trainer.evaluate()
        print("Validation metrics:", metrics)

        trainer.save_model(OUTPUT_ROOT / "single_run")
        tokenizer.save_pretrained(OUTPUT_ROOT / "single_run")


if __name__ == "__main__":
    main()

