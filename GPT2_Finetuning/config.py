# config.py
from pathlib import Path

###############################################################
#     Centralized Path Specs Related To Fine Tuning GPT2      #
###############################################################

FT_GPT2_BASE_DIR = Path(__file__).resolve().parent
FT_GPT2_OUTPUT_ROOT = FT_GPT2_BASE_DIR / "gpt2-finetune-outcome"
FT_GPT2_BEST_PARAMS_FILE = FT_GPT2_OUTPUT_ROOT / "best_trial_params.json"
FT_GPT2_BEST_METRICS_FILE = FT_GPT2_OUTPUT_ROOT / "best_trial_metrics.json"
FT_GPT2_ALL_TRIALS_FILE = FT_GPT2_OUTPUT_ROOT / "all_trials.jsonl"
FT_GPT2_BEST_MODEL_DIR = FT_GPT2_OUTPUT_ROOT / "best_model"

FT_GPT2_LORA_OUTPUT_ROOT = FT_GPT2_BASE_DIR / "lora_gpt2_finetune_outcome"
FT_GPT2_LORA_BEST_PARAMS_FILE = FT_GPT2_LORA_OUTPUT_ROOT / "best_trial_params.json"
FT_GPT2_LORA_BEST_METRICS_FILE = FT_GPT2_LORA_OUTPUT_ROOT / "best_trial_metrics.json"
FT_GPT2_LORA_ALL_TRIALS_FILE = FT_GPT2_LORA_OUTPUT_ROOT / "all_trials.jsonl"
FT_GPT2_LORA_BEST_MODEL_DIR = FT_GPT2_LORA_OUTPUT_ROOT / "best_model"

#############################################################
#   Centralized Path Specs For Zero/Few-Shot Inferencing    #
#############################################################

ZSHOT_BASE_DIR = FT_GPT2_BASE_DIR / "ZeroShot_FewShot" / "ZeroShot"
FSHOT_BASE_DIR = FT_GPT2_BASE_DIR / "ZeroShot_FewShot" / "FewShot"

ZSHOT_PATHS = {
    "GPT2": ZSHOT_BASE_DIR / "GPT2_Results",
    "BART": ZSHOT_BASE_DIR / "BART_Results",
    "PEGASUS": ZSHOT_BASE_DIR / "Pegasus_Results",
}

FSHOT_GPT2_RES = FSHOT_BASE_DIR / "GPT2_Results"


print(FT_GPT2_BASE_DIR)
