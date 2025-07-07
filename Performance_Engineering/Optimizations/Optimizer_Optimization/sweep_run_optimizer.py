#!/usr/bin/env python3
"""
sweep_optimizers.py  –  v3 (OOM-hardened)

Changes vs v2
-------------
✓ Driver no longer imports torch  → no lingering CUDA context.
✓ nvidia-smi used for VRAM stats.
✓ Stricter orphan killer: matches python, torchrun, deepspeed.
✓ Drop CPython malloc arenas with `multiprocessing.heap`.
✓ Optional page-cache flush (`sudo` required) via --drop-cache.
"""

import argparse, copy, os, shutil, subprocess, sys, time, gc, yaml, re, psutil

DEFAULT_OPTIMISERS = [
    "adafactor", "adamw", "adamp", "qhm", "sophiag",
    "adam8bit", "lion8bit"
]

# ----------------------------- helpers ---------------------------------- #
def load_template(path: str) -> dict:
    with open(path, "r") as fh:
        tpl = yaml.safe_load(fh)
    if not isinstance(tpl, list) or len(tpl) != 1:
        sys.exit(f"[FATAL] {path} must be a single-item YAML list]")
    return tpl[0]

def generate_cfg(base: dict, opt: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["optimizer"] = opt
    cfg["name"] = f"{cfg['model_name']}_{opt}_{cfg['batch_size']}"

    # Only enable DeepSpeed for fusedadam
    if opt.lower() == "fusedadam":
        cfg["use_deepspeed"] = True
    else:
        cfg.pop("use_deepspeed", None)  # Explicitly remove if inherited from base

    return cfg

def write_cfg(cfg: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"config_{cfg['optimizer']}.yaml")
    with open(p, "w") as fh:
        yaml.safe_dump([cfg], fh, sort_keys=False)
    return p

def run_benchmark(cfg: str, dataset: str, steps: int, csv: str) -> int:
    cmd = [
        sys.executable, "profile_performance_gpt2.py",
        "--dataset", dataset, "--configs", cfg,
        "--steps", str(steps), "--csv", csv,
    ]

    print("└─ running:", " ".join(cmd))
    return subprocess.call(cmd)

# ---------------- VRAM / RAM hygiene ------------------------------------ #
_NVSMI_RE = re.compile(r"^\s*(\d+)\s*$")
def vram_free_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]
    ).decode().splitlines()[0]
    m = _NVSMI_RE.match(out)
    return int(m.group(1)) if m else -1

_ZOMBIE_PAT = re.compile(r"(profile_performance_gpt2|torchrun|deepspeed)")
def kill_orphans() -> None:
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if _ZOMBIE_PAT.search(" ".join(p.info["cmdline"])):
                print(f"  ‣ kill orphan pid={p.pid}")
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def drop_cpython_arenas() -> None:
    # frees huge arenas that gc.collect doesn't touch
    import _ctypes, ctypes
    libc = ctypes.CDLL(None)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)

def flush_page_cache() -> None:
    subprocess.call(["sudo", "sync"])
    subprocess.call(
        ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

# ------------------------------- main ------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--template", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output-dir", default="generated_cfgs")
    ap.add_argument("--optimizers", nargs="+", default=DEFAULT_OPTIMISERS)
    ap.add_argument("--wait", type=int, default=20)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--drop-cache", action="store_true",
                    help="Flush Linux page cache between runs (needs sudo).")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)

    base = load_template(args.template)
    failures = []

    for opt in args.optimizers:
        cfg_path = write_cfg(generate_cfg(base, opt), args.output_dir)

        print(f"\n=== {opt} | free VRAM before: {vram_free_mb():,} MB ===")
        rc = run_benchmark(cfg_path, args.dataset, args.steps, args.csv)
        if rc != 0:
            failures.append(opt)

        # ---- aggressive cleanup ----------------------------------------- #
        kill_orphans()          # stray processes
        gc.collect()            # Python objects
        drop_cpython_arenas()   # huge arenas
        if args.drop_cache:
            flush_page_cache()  # OS page cache (sudo)
        # ----------------------------------------------------------------- #

        print(f"=== {opt} | free VRAM after:  {vram_free_mb():,} MB ===")
        print(f"sleeping {args.wait}s …")
        time.sleep(args.wait)

    if failures:
        print("\n Failed runs:", ", ".join(failures))
        sys.exit(1)
    print("\n Sweep finished cleanly.")

if __name__ == "__main__":
    main()
