"""
  python generate_and_evaluate.py \
  --model-path GroundUp_ModelTraining_Outcome/gpt2_samsum.pt \
  --config configs/samsum.yaml \
  --dataset samsum \
  --split test \
  --line 42
"""

import argparse
import os
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from model import GPT
from evaluation import evaluate_model
from generate import generate_summary_on_input


def load_model(model_path, config):
    print(f"\nLoading model from: {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(config["tokenizer"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model_cfg = {
        "vocab_size": tokenizer.vocab_size,
        **config["model"]
    }
    model = GPT(model_cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, tokenizer


def run_generation(model, tokenizer, dataset, line_num):
    if line_num >= len(dataset):
        raise IndexError(f"Line number {line_num} out of range. Dataset has {len(dataset)} entries.")

    example = dataset[line_num]
    if "dialog" in example:
        input_text = " ".join(example["dialog"])
    elif "dialogue" in example:
        input_text = example["dialogue"]
    else:
        raise KeyError("Dataset does not contain 'dialogue' or 'dialog' fields.")

    summary = generate_summary_on_input(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        max_len=50,
        k=5,
        temperature=1.0
    )

    print("\n===== INPUT =====")
    print(input_text)
    print("\n===== GENERATED SUMMARY =====")
    print(summary)
    if "summary" in example:
        print("\n===== REFERENCE SUMMARY =====")
        print(example["summary"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained model .pt file")
    parser.add_argument("--config", required=True, help="Path to YAML config used during training")
    parser.add_argument("--dataset", default="samsum", help="Hugging Face dataset name")
    parser.add_argument("--split", default="test", help="Which split to evaluate/generate from")
    parser.add_argument("--limit", type=int, default=100, help="How many samples to use for ROUGE evaluation")
    parser.add_argument("--line", type=int, help="Evaluate model generation on this specific line number")
    parser.add_argument("--output-to", type=str, help="Optional path to save evaluation or generation output")

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_model(args.model_path, config)
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    output_lines = []

    if args.line is not None:
        run_generation(model, tokenizer, dataset, args.line)

    else:
        print(f"\nRunning ROUGE evaluation on {args.limit} examples from {args.dataset}/{args.split}...")
        scores = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_len=config["model"]["context_length"],
            k=5,
            temperature=1.0,
            limit=args.limit,
            device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("\n===== ROUGE SCORES =====")
        for key, val in scores.items():
            print(f"{key:>10}: {val:.4f}")
        if args.output_to:
            output_lines.append("\n===== ROUGE SCORES =====")
            output_lines += [f"{key}: {val:.4f}" for key, val in scores.items()]

    if args.output_to and output_lines:
        os.makedirs(os.path.dirname(args.output_to), exist_ok=True)
        with open(args.output_to, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\n Output saved to: {args.output_to}")


if __name__ == "__main__":
    main()
