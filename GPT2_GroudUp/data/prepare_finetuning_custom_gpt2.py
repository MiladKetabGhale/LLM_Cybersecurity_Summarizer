#!/usr/bin/env python3
"""
Load **train.jsonl / validation.jsonl** containing
{"source": "...", "summary": "..."}
and emit two PyTorch tensors:
  • input_ids  – prompt + target
  • labels     – −100 for prompt/pad tokens, ids for target tokens
"""
from pathlib import Path
import json
import torch
from datasets import Dataset
from transformers import GPT2Tokenizer


# ---- helper --------------------------------------------------------------
def _tokenize_examples(batch, tok: GPT2Tokenizer, block, pad_id,
                       min_sum_tokens=25):
    outs_ids, outs_lbls = [], []

    for src, tgt in zip(batch["source"], batch["summary"]):
        tgt_ids = tok(tgt, add_special_tokens=False)["input_ids"]

        # --------------------------------------------------------------- #
        # build the prompt AFTER any potential truncation so the           #
        # "one-third of N tokens" message is accurate                      #
        # --------------------------------------------------------------- #
        def build_prompt(tgt_len):                                        ### fixed
            return (                                                      ### fixed
                f"<|summarize|> Document: {src}\n\n"                      ### fixed
                f"### Task: Summarise the document for a cybersecurity "  ### fixed
                f"analyst within 25 to 75 tokens.\n\n"  ### fixed
                f"### Summary: "                                          ### fixed
            )                                                             ### fixed
        # ---------------------------------------------------------------- #

        prompt_ids = tok(build_prompt(len(tgt_ids)), add_special_tokens=False)["input_ids"]

        # ── guarantee at least `min_sum_tokens` survive ────────────────
        if len(prompt_ids) + len(tgt_ids) > block:
            keep_pr = block - len(tgt_ids)
            if keep_pr < 1:                      # prompt alone overflows
                keep_pr = block - min_sum_tokens
                tgt_ids = tgt_ids[:min_sum_tokens]
            prompt_ids = prompt_ids[:keep_pr]

            # rebuild prompt to keep “one-third” line truthful            ### fixed
            prompt_ids = tok(                                             ### fixed
                build_prompt(len(tgt_ids)),                               ### fixed
                add_special_tokens=False                                  ### fixed
            )["input_ids"][:keep_pr]                                      ### fixed

        ids  = prompt_ids + tgt_ids
        pad  = [pad_id] * (block - len(ids))
        outs_ids.append(ids + pad)

        lbl = [-100] * len(prompt_ids) + tgt_ids
        lbl = lbl + [-100] * (block - len(lbl))
        outs_lbls.append(lbl)

    return {"input_ids": outs_ids, "labels": outs_lbls}


# ---- main entry ----------------------------------------------------------
def load_and_prepare_dataset(cfg):
    tok = GPT2Tokenizer.from_pretrained(cfg["tokenizer"]["name"])
    tok.pad_token = tok.eos_token
    pad_id     = tok.pad_token_id
    block_size = cfg["data"]["max_length"]

    root = Path(cfg["data"]["root"])
    files = {
        "train": root / "train.jsonl",
        "validation": root / "validation.jsonl",
    }
    ds = {split: Dataset.from_json(str(path)) for split, path in files.items()}

    # batched tokenisation
    for split in ds:
        ds[split] = ds[split].map(
            _tokenize_examples,
            fn_kwargs=dict(tok=tok, block=block_size, pad_id=pad_id),
            batched=True,
            remove_columns=ds[split].column_names,
        )
        ds[split].set_format(type="torch")

    train_tensor = torch.stack(                             ### fixed
        [ex["input_ids"] for ex in ds["train"]]             ### fixed
    )      
    val_tensor = torch.stack(                               ### fixed
        [ex["input_ids"] for ex in ds["validation"]]        ### fixed
    )   
    lbl_train = torch.stack(                                ### fixed
        [ex["labels"] for ex in ds["train"]]                ### fixed
    )
    lbl_val = torch.stack(                                  ### fixed
        [ex["labels"] for ex in ds["validation"]]           ### fixed
    )
    return (train_tensor, lbl_train), (val_tensor, lbl_val)

