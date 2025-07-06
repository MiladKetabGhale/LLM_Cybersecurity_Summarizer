#!/usr/bin/env python3
"""
run_benchmark_sweep.py
Launches benchmark_performance_gpt2_v2.py for every Cartesian combination
defined in a sweep_config.yaml.

Running the script:
python run_benchmark_sweep.py \
    --sweep_config sweep_config.yaml \
    --dataset ../../DataPreprocessing/train.jsonl \
    --output_csv results.csv \  
    --delay 40
"""

import argparse
import os
import yaml
import itertools
import subprocess
import time
import gc
from tempfile import NamedTemporaryFile
from math import prod

import torch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_sweep_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cartesian_product(config_dict):
    """
    Yields each configuration as a dict.
    Scalars are treated as 1-element lists to avoid string-is-iterable bugs.
    """
    keys = list(config_dict.keys())
    values = [
        v if isinstance(v, (list, tuple)) else [v]
        for v in config_dict.values()
    ]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# Short aliases to keep names readable
_MODE_ALIAS = {
    "default": "def",
    "reduce-overhead": "ro",
    "max-autotune": "ma",
}


def format_run_name(cfg):
    """
    Build a human-readable, mostly unique string.
    Example: gpt2_fp16_T_ro_def_2
    """
    parts = [
        cfg.get("model_name", "model"),
        cfg.get("precision", "fp32"),
        "T" if cfg.get("compile", False) else "F",
        _MODE_ALIAS.get(cfg.get("compile_mode", "def"), cfg.get("compile_mode")),
        cfg.get("attn_backend", "def"),
        str(cfg.get("batch_size", "bs")),
    ]
    return "_".join(str(p) for p in parts)


def run_benchmark_for_cfg(
    cfg,
    dataset_path,
    benchmark_script,
    output_csv,
    steps,
    delay_sec,
):
    cfg["name"] = format_run_name(cfg)

    with NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
        yaml.dump([cfg], tmp)
        tmp_path = tmp.name

    print(f"\n Running: {cfg['name']}")

    try:
        result = subprocess.run(
            [
                "python",
                benchmark_script,
                "--dataset",
                dataset_path,
                "--configs",
                tmp_path,
                "--csv",
                output_csv,
                "--steps",
                str(steps),
            ],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed: {cfg['name']}  (return code {result.returncode})")

    finally:
        os.remove(tmp_path)

        # --- Cleanup ---
        torch.cuda.empty_cache()
        gc.collect()

        if delay_sec > 0:
            print(f"⏳ Waiting {delay_sec} s before next run…\n")
            time.sleep(delay_sec)


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config", required=True,
                        help="YAML with parameter ranges")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_csv", default="sweep_results.csv")
    parser.add_argument("--benchmark_script",
                        default="benchmark_performance_gpt2_v2.py")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--delay", type=int, default=10,
                        help="Delay (s) between runs")
    args = parser.parse_args()

    sweep_cfg = load_sweep_config(args.sweep_config)

    # Informative: how many runs are we about to launch?
    total_runs = prod(
        len(v if isinstance(v, (list, tuple)) else [v])
        for v in sweep_cfg.values()
    )
    print(f"Total planned runs: {total_runs}")

    for cfg in cartesian_product(sweep_cfg):
        # Example guard: skip meaningless combos
        if not cfg.get("compile", False) and cfg.get("compile_mode") not in (None, "default"):
            print(f"Skipping {cfg} (compile_mode meaningless without compile=True)")
            continue

        run_benchmark_for_cfg(
            cfg,
            args.dataset,
            args.benchmark_script,
            args.output_csv,
            args.steps,
            args.delay,
        )


if __name__ == "__main__":
    main()
