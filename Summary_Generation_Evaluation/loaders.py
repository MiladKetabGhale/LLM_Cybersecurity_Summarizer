import json
from pathlib import Path
from typing import Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

def load_jsonl(path: Union[str, Path]):
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        # try sibling-of-current-folder fallback
        alt = Path(__file__).resolve().parent.parent / path
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_decoder_only(model_key: str) -> bool:
    """Return True for GPT‑2‑style models."""
    return model_key in {"gpt2_local", "gpt2_ft", "gpt2_ft_lora"}


def load_tokenizer(model_key: str, model_dir: Union[str, Path, None]):
    if is_decoder_only(model_key):
        tok = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
        tok.pad_token = tok.eos_token
        return tok
    return AutoTokenizer.from_pretrained(model_key)


def load_model(model_key: str, model_dir: Union[str, Path, None], device):
    if is_decoder_only(model_key):
        model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_key)
    return model.to(device).eval()

# loaders.py  (append just before the end of the file)

def load_model_and_tok(cfg, device):
    """
    Decide where to load the model & tokenizer from based on:
      cfg["mode"]      : "fine-tuned" | "few-shot" | "zero-shot" (default)
      cfg.get("model") : HF hub id or custom tag
      cfg.get("model_dir") : local weights path (optional)
    """
    mode       = cfg.get("mode", "zero-shot")
    model_key  = cfg.get("model")
    model_dir  = cfg.get("model_dir")        # may be None

    # Fine-tuned must supply a local dir
    if mode == "fine-tuned" and not model_dir:
        raise ValueError("mode=fine-tuned requires `model_dir` with saved weights")

    # Prefer local_dir if given (works for any mode)
    if model_dir:
        tok   = load_tokenizer(model_key, model_dir)
        model = load_model(model_key, model_dir, device)
        return tok, model

    # Otherwise we need a HF id
    if not model_key:
        raise ValueError("Neither `model_dir` nor `model` id provided in config")

    tok   = load_tokenizer(model_key, None)
    model = load_model(model_key, None, device)
    return tok, model
