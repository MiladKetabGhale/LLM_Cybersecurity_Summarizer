"""The main entry point for running the module for generation and evaluation of various models
     • hand‑crafted (decoder‑only) GPT‑2  ("gpt2_local")
     • fine‑tuned GPT‑2                ("gpt2_ft")
     • HuggingFace zero‑shot models:   ("bart-base", "pegasus-xsum",
                                        "distilbart-cnn-12-6")
   ------------------------------------------------------------------------------
   Expected JSONL schema  : {"source": ..., "summary": ...}
   Config YAML structure  :
      model:              bart-base           # or gpt2_local | gpt2_ft | ...
      model_dir:          models/my_gpt2      # required for the two local GPT‑2s
      data_file:          DataPreprocessing/validation.jsonl
      lines:              5                   # int or [5, 12, 20]
      max_new_tokens:     128
      evaluation_metric:  rouge               # rouge | bleu | meteor
  
  Run:  python main.py --config confis/<yaml file you want to evaluate its model>
  """

from __future__ import annotations
from datetime import datetime
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from config_loader import load_config
from loaders import get_device, load_tokenizer, load_model, load_jsonl, is_decoder_only
from prompts import gpt2_prompt
from generation import generate_decoder, generate_seq2seq
from evaluation import score
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Union

# ─────────────────────────────────────────────────────────────────────────────── #
# FUNCTION: run                                                                   #
#                                                                                 #
# This is the **main execution function** called by CLI or another script.        #
# It loads the model and tokenizer, prepares evaluation data,                     #
# handles prompt generation, model inference, evaluation scoring,                 #
# and saves outputs per sample + optional aggregate scores.                       #
#                                                                                 #
# Key responsibilities:                                                           #
#  - Load model/tokenizer from config                                             #
#  - Evaluate given lines or whole dataset                                        #
#  - Generate summaries depending on decoder-only or encoder-decoder architecture #
#  - Save results and print per-line/average scores                               #
# ─────────────────────────────────────────────────────────────────────────────── #

def run(cfg: Dict[str, Any]):
    device = get_device()
    model_key = cfg["model"]
    model_dir = cfg.get("model_dir")  # may be None for HF models
    data_file = Path(cfg["data_file"])
    data = load_jsonl(data_file)

    # ─── Normalize lines input ───────────────────────────────────────────────
    lines = cfg["lines"]
    if isinstance(lines, int):
        lines = [lines]
    elif lines == [0]:
        raise ValueError("Invalid value for 'lines': [0] — To evaluate all lines, use an empty list: []")
    elif isinstance(lines, list) and len(lines) == 0:
        lines = list(range(1, len(data) + 1))  # Evaluate all lines

    # ─── Load model/tokenizer ────────────────────────────────────────────────
    tok = load_tokenizer(model_key, model_dir)
    model = load_model(model_key, model_dir, device)

    # ─── Output path construction ────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg["lines"] == []:
        datafile_name = data_file.stem
        out_file = output_dir / f"{model_key}_{datafile_name}_{timestamp}.jsonl"
    else:
        out_file = output_dir / f"{model_key}_{timestamp}.jsonl"

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # ─── Evaluation and Save ─────────────────────────────────────────────────
    aggregated_scores = {}
    with out_file.open("w", encoding="utf-8") as f:
        for ln in lines:
            if ln < 1 or ln > len(data):
                print(f"[WARN] line {ln} out of range (1‥{len(data)}) — skipped")
                continue

            sample = data[ln - 1]
            source, gold = sample["source"], sample["summary"]

            inference_time_ms = None  # Predeclare for safety
 
            if is_decoder_only(model_key):
                prompt = gpt2_prompt(source)
                inputs = tok(prompt, return_tensors="pt").to(device)
            else:
                inputs = tok(source, return_tensors="pt").to(device)
 
            start = time.time()
            output_ids = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"])
            end = time.time()
 
            inference_time_ms = (end - start) * 1000
            pred = tok.decode(output_ids[0], skip_special_tokens=True)
 
            # Token counts
            input_tokens = inputs["input_ids"].shape[1]
            if is_decoder_only(model_key):
                output_tokens = output_ids.shape[1] - input_tokens
            else:
                output_tokens = output_ids.shape[1]
             

            print("\n──────── LINE", ln, "────────")
            print("SOURCE:", source)
            print("GOLD  :", gold)
            print("PRED  :", pred)

            scores = score(pred, gold, cfg["evaluation_metric"])
            print("SCORES:")
           
            for k, v in scores.items():
                print(f"  {k:<8}: {v:.4f}")
                aggregated_scores.setdefault(k, []).append(v)

            json.dump({
                "line": ln,
                "model": model_key,
                "scores": scores,
                "inference_time_ms": round(inference_time_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "time_per_token_ms": round(inference_time_ms / max(output_tokens, 1), 2)
            }, f)
            f.write("\n")

        # ─── Append average if evaluating whole file ─────────────────────────
        if cfg["lines"] == []:
            avg_scores = {
                k: sum(v_list) / len(v_list) for k, v_list in aggregated_scores.items()
            }
            json.dump({
                "line": "average",
                "model": model_key,
                "scores": avg_scores
            }, f)
            f.write("\n")
            print("\n──────── AVERAGE SCORES ────────")
            for k, v in avg_scores.items():
                print(f"  {k:<8}: {v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────── #
# CLI ENTRYPOINT                                                                  #
#                                                                                 #
# Defines a command-line parser to specify config YAML file path.                 #
# Loads the YAML config and invokes the main run() function.                      #
# Allows easy invocation: `python summariser_unified.py --config summariser.yaml` #
# ─────────────────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="Universal summariser generator & evaluator")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg)

