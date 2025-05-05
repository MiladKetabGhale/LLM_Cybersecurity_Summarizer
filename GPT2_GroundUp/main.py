import sys
import os
import argparse                    # FIX – use argparse for named flags
import matplotlib
import matplotlib.pyplot as plt
import importlib.util
import torch
import resource
from transformers import GPT2Tokenizer
from transformers import get_cosine_schedule_with_warmup
from config_utils import load_config
from model import GPT
from profile_train import train_summarizer

matplotlib.use("Agg")  # backend without display

# ──────────────────────────────────────────────────────────────────────────────── #
# HELPER FUNCTIONS                                                                 #
#                                                                                  #
# Define utility functions to:                                                     #
# 1. Increase OS open file limits if needed                                        #
# 2. Dynamically import a preprocessing script that must expose                    #
#    `load_and_prepare_dataset(config)` function                                   #
# ──────────────────────────────────────────────────────────────────────────────── #

def set_max_open_files(min_soft_limit=2048):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < min_soft_limit:
        new_soft = min(min_soft_limit, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"Increased open file limit to: {new_soft}")
    else:
        print(f"Open file limit is sufficient: {soft}")

def load_prepare_function(script_path):
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"[ERROR] Preprocessing script not found: {script_path}")

    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_and_prepare_dataset"):
        raise AttributeError(
            f"[ERROR] Script {script_path} must define 'load_and_prepare_dataset(config)'"
        )
    return module.load_and_prepare_dataset

# ──────────────────────────────────────────────────────────────────────────────── #
# MAIN SCRIPT ENTRY POINT                                                          #
#                                                                                  #
# Parses command-line arguments for config and preprocessing paths.                #
# Initializes tokenizer, model, data, optimizer, scheduler.                        #
# Starts training loop and saves the model + learning curve plot.                  #
#                                                                                  #
# This is the main controller script for end-to-end model training.                #
# ──────────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    
    # ──────────────────────────────────────────────────────────────────────────── #
    # ARGUMENT PARSING                                                             #
    # Parse command-line arguments for config file and preprocessing script paths. #
    # Enables easy CLI usage instead of hardcoded paths.                           #
    # ──────────────────────────────────────────────────────────────────────────── #
    parser = argparse.ArgumentParser(
        description="Train tiny GPT model with named arguments."
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--prep",
        "-p",
        required=True,
        help="Path to Python preprocessing script implementing load_and_prepare_dataset()",
    )
    args = parser.parse_args()

    config_path      = args.config      # FIX
    prep_script_path = args.prep        # FIX

    # ──────────────────────────────────────────────────────────────────────────── #
    # INITIALIZATION                                                               #
    # Increase open file limits (macOS/Linux) and load configuration YAML.         #
    # Select computation device (MPS → CUDA → CPU).                                #
    # Load tokenizer and vocab size from HuggingFace.                              #
    # ──────────────────────────────────────────────────────────────────────────── #

    set_max_open_files()
    print("Loading config...")
    config = load_config(config_path)

    # device choice unchanged
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Getting vocab size from tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(config["tokenizer"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    # ──────────────────────────────────────────────────────────────────────────── #
    # DATA LOADING                                                                 #
    # Dynamically import and call the dataset preparation function.                #
    # Should return preprocessed train/validation tensors compatible with model.   #
    # ──────────────────────────────────────────────────────────────────────────── #

    print(f"Loading dataset using: {prep_script_path} ...")
    load_and_prepare_dataset = load_prepare_function(prep_script_path)
    train_tensors, validation_data = load_and_prepare_dataset(config)

    print("Creating model...")
    model_cfg = {"vocab_size": tokenizer.vocab_size, **config["model"]}
    model = GPT(model_cfg)

    # ──────────────────────────────────────────────────────────────────────────── #
    # DATALOADER CREATION                                                          #
    # Construct PyTorch DataLoaders for training and validation datasets.          #
    # Batch size, shuffle, workers configurable via YAML.                          #
    # ──────────────────────────────────────────────────────────────────────────── #

    print("\n Creating training dataloader...")
    train_loader = torch.utils.data.DataLoader(
        train_tensors,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
    )

    # ── optimiser & scheduler ───────────────────────────────────────────
    print("\n Starting the training process")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.01),
    )

    total_steps  = len(train_loader) * config["train"]["num_epochs"]
    warmup_steps = int(config["train"].get("warmup_ratio", 0.06) * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # FIX – added intra-epoch early-stop args
    train_curve, val_curve = train_summarizer(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config["train"]["num_epochs"],
        patience=config["train"].get("patience", 4),
        val_every_steps=config["train"].get("val_every_steps", 8000),  # FIX
        step_patience=config["train"].get("step_patience", 4),           # FIX
        val_subset=config["train"].get("val_subset", 0.2),               # FIX
    )

    # ── save model & plot curves ────────────────────────────────────────
    model_output_path = config["train"].get("model_output_path", None)
    if model_output_path is None:
        dataset_name = config["data"]["dataset_name"]
        os.makedirs("GroundUp_ModelTraining_Outcome", exist_ok=True)
        model_output_path = f"GroundUp_ModelTraining_Outcome/gpt2_{dataset_name}.pt"

    torch.save(model.state_dict(), model_output_path)
    print(f"\n Trained model saved to: {model_output_path}")

    plt.figure()
    plt.plot(train_curve, label="train")
    plt.plot(val_curve,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve")
    plt.legend()
    out_png = os.path.splitext(model_output_path)[0] + ".png"
    plt.savefig(out_png)
    print(f"Learning-curve plot saved → {out_png}")
