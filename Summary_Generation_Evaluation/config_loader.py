from pathlib import Path
from typing import Dict, Any
import yaml

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    required = [
        "model",
        "data_file",
        "lines",
        "max_new_tokens",
        "evaluation_metric",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required key in config: {key}")

    # Normalise
    if isinstance(cfg["lines"], int):
        cfg["lines"] = [cfg["lines"]]
    cfg["model"] = str(cfg["model"]).lower()
    cfg["evaluation_metric"] = str(cfg["evaluation_metric"]).lower()

    return cfg

