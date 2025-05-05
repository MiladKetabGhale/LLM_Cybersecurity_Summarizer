# visualize_trials.py
import json
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FT_GPT2_ALL_TRIALS_FILE

# 1) Load JSONL into pandas
records = []
with open(ALL_TRIALS_PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        records.append(json.loads(line))
df = pd.json_normalize(records)

# df columns include:
#  - number
#  - value  (this is your eval_loss)
#  - params.learning_rate
#  - params.weight_decay
#  - params.num_train_epochs
#  - params.per_device_train_batch_size
#  - params.gradient_accumulation_steps
#  - metrics.eval_loss  (same as value)
#  - metrics.eval_samples_per_second, etc.

# 2) Plot eval_loss vs trial number
plt.figure(figsize=(8,4))
plt.plot(df["number"], df["value"], marker="o", linestyle="-")
plt.xlabel("Trial #")
plt.ylabel("Eval Loss")
plt.title("Optuna Optimization History")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Plot eval_loss vs learning_rate (log scale)
plt.figure(figsize=(8,4))
plt.scatter(df["params.learning_rate"], df["value"])
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Eval Loss")
plt.title("Learning Rate vs Eval Loss")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

