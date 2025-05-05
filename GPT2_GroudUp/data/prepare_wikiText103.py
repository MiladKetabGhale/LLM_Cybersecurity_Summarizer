# prepare_wikiText103.py
"""
Tokenise WikiText-103 (or any HF dataset) and chunk it for GPT training.

Changes vs. the old version
---------------------------
1. **NO hard 50 000-sample cap** – the full corpus is used by default.
2. Optional `samples_per_epoch` can be set in the YAML
   (under `data:`) if you *do* want a cap.
3. All splits (train / val / test) are cached to
   `data/wikiText_preprocessed/*.pt` to avoid re-tokenising next run.
4. Basic stats printed so you know how many tokens / chunks you got.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

# ── where we cache the pre-tokenised tensors ──────────────────────────────
PREP_DIR = Path(__file__).resolve().parent.parent / "data" / "wikiText_preprocessed"
PREP_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────
def _chunkify(token_ids: torch.Tensor, max_len: int, stride: int) -> torch.Tensor:
    """Return a [N, max_len] tensor with (optionally overlapping) chunks."""
    chunks = [
        token_ids[i : i + max_len]
        for i in range(0, len(token_ids) - max_len + 1, stride)
    ]
    # ragged tail so we never drop text
    if len(token_ids) % stride != 0 and len(token_ids) >= max_len:
        chunks.append(token_ids[-max_len:])
    return torch.stack(chunks)


def _encode_corpus(dataset_split, tokenizer) -> torch.Tensor:
    ids = []
    for ex in dataset_split:
        txt = ex["text"].strip()
        if txt:
            ids.extend(tokenizer.encode(txt + tokenizer.eos_token))
    return torch.tensor(ids, dtype=torch.long)


# ── public entry point (called by main.py) ────────────────────────────────
def load_and_prepare_dataset(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (train_chunks, val_chunks).  Also saves *.pt files into PREP_DIR."""
    tokenizer = GPT2Tokenizer.from_pretrained(config["tokenizer"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(
        config["data"]["dataset_name"], config["data"]["dataset_config"]
    )

    max_len  = int(config["data"]["max_length"])
    stride   = int(config["data"]["stride"])
    cap      = config["data"].get("samples_per_epoch")      # None → no cap

    # ── TRAIN ────────────────────────────────────────────────────────────
    train_ids     = _encode_corpus(ds["train"], tokenizer)
    train_chunks  = _chunkify(train_ids, max_len, stride)

    if cap:                                                # optional down-sampling
        if len(train_chunks) > cap:
            perm = torch.randperm(len(train_chunks))[:cap]
            train_chunks = train_chunks[perm]
    torch.save(train_chunks, PREP_DIR / "train.pt")

    # ── VALIDATION ───────────────────────────────────────────────────────
    val_ids    = _encode_corpus(ds["validation"], tokenizer)
    val_chunks = _chunkify(val_ids, max_len, stride)
    torch.save(val_chunks, PREP_DIR / "validation.pt")

    # ── TEST (optional split) ────────────────────────────────────────────
    if "test" in ds:
        test_ids    = _encode_corpus(ds["test"], tokenizer)
        test_chunks = _chunkify(test_ids, max_len, stride)
        torch.save(test_chunks, PREP_DIR / "test.pt")

    # ── simple stats ─────────────────────────────────────────────────────
    print(f"[STATS] train  : {len(train_chunks):,} chunks  "
          f"({len(train_chunks)*max_len:,} tokens)")
    print(f"[STATS] val    : {len(val_chunks):,} chunks  "
          f"({len(val_chunks)*max_len:,} tokens)")
    if (PREP_DIR / "test.pt").exists():
        print(f"[STATS] test   : {len(test_chunks):,} chunks")

    return train_chunks, val_chunks
