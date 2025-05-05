#!/usr/bin/env python3
"""
generate_and_evaluate.py

 - Loads an already-trained GPT-style summariser (.pt)  
 - Generates or evaluates summaries on a JSON/JSONL dataset  
 - Prints **source / gold / predicted** for every evaluated example  
 - Saves the same text (plus ROUGE scores) to --output-to if provided

Minimal patch over the original script:
* Works with arbitrary field names via --text-field / --summary-field
* Gracefully handles datasets that miss some keys, never crashes on n==0
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer

# ---- local imports ---------------------------------------------------------
from model import GPT
from generate import generate_summary_topk_temperature

# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, config: dict):
    print(f"\nLoading model from: {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(config["tokenizer"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model_cfg = {"vocab_size": tokenizer.vocab_size, **config["model"]}
    model = GPT(model_cfg)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, max_len: int, k=5, temperature=1.0, device="cpu"):
    """Generate summary string given a prompt (includes TL;DR sentinel)."""
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out_ids = generate_summary_topk_temperature(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=max_len,
        k=k,
        temperature=temperature,
        device=device,
        end_token_id=tokenizer.eos_token_id,
    )
    out_text = tokenizer.decode(out_ids[0].tolist())
    # keep only text after TL;DR:
    return out_text.split("TL;DR:")[-1].strip().replace("<|endoftext|>", "")


def evaluate_and_log(
    model,
    tokenizer,
    dataset,
    text_field: str,
    summary_field: str,
    max_len: int,
    limit: int,
    device: str,
    k: int = 5,
    temperature: float = 1.0,
):
    """Loop over dataset, print & collect outputs, return average ROUGE dict + log lines."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0
    logs = []

    for ex in dataset:
        if n >= limit:
            break

        source = ex.get(text_field) or ex.get("dialogue") or ex.get("dialog") or ""
        gold = ex.get(summary_field) or ex.get("summary") or ""
        if not source or not gold:
            continue  # skip incomplete rows

        prompt = f"Summarize: {source.strip()} TL;DR:"
        pred = generate_one(model, tokenizer, prompt, max_len, k, temperature, device)

        # ROUGE
        scores = scorer.score(gold, pred)
        for k_ in totals:
            totals[k_] += scores[k_].fmeasure

        # console + log file
        print(f"\n—— Example {n+1} ——")
        print("SOURCE :", source)
        print("GOLD   :", gold)
        print("PRED   :", pred)

        logs.extend(
            [
                f"—— Example {n+1} ——",
                f"SOURCE : {source}",
                f"GOLD   : {gold}",
                f"PRED   : {pred}",
                "",
            ]
        )
        n += 1

    if n == 0:
        print("[ERROR] No valid evaluation examples found (check field names).")
        return None, logs

    # average ROUGE
    avg = {k: v / n for k, v in totals.items()}
    print(f"\n📊 ROUGE on {n} examples:")
    for k, v in avg.items():
        print(f"{k:<8}: {v:.4f}")

    logs.append("📊 AVERAGE ROUGE")
    logs.extend([f"{k}: {v:.4f}" for k, v in avg.items()])

    return avg, logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--config", required=True, help="YAML config used during training")
    parser.add_argument("--dataset", required=True, help="Path or HF dataset name")
    parser.add_argument("--split", default="test", help="Dataset split to use")
    parser.add_argument("--limit", type=int, default=100, help="#examples for evaluation")
    parser.add_argument("--line", type=int, help="Just run generation on this line #")
    parser.add_argument("--output-to", help="Where to write logs / scores")
    # new minimal-impact flags
    parser.add_argument("--text-field", default="source", help="Field with source text")
    parser.add_argument("--summary-field", default="summary", help="Field with gold summary")

    args = parser.parse_args()

    # load YAML
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model(args.model_path, cfg)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = load_dataset("json", data_files=args.dataset, split=args.split)

    if args.line is not None:
        if args.line >= len(dataset):
            raise IndexError(f"--line {args.line} out of range (dataset has {len(dataset)} rows)")
        ex = dataset[args.line]
        src = ex.get(args.text_field) or ex.get("dialogue") or ex.get("dialog") or ""
        prompt = f"Summarize: {src.strip()} TL;DR:"
        pred = generate_one(model, tokenizer, prompt, cfg["model"]["context_length"], device=device)
        print("\n===== INPUT =====\n", src)
        print("\n===== GENERATED SUMMARY =====\n", pred)
        if "summary" in ex or args.summary_field in ex:
            print("\n===== REFERENCE SUMMARY =====\n", ex.get(args.summary_field, ex.get("summary", "")))
        return

    # evaluation mode
    rouge_scores, log_lines = evaluate_and_log(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        text_field=args.text_field,
        summary_field=args.summary_field,
        max_len=cfg["model"]["context_length"],
        limit=args.limit,
        device=device,
    )

    if args.output_to and log_lines:
        out_path = Path(args.output_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"\n📝 Output written to: {out_path}")


if __name__ == "__main__":
    main()

