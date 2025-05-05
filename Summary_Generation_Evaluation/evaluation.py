import evaluate
from typing import Dict

def score(pred: str, ref: str, metric: str) -> Dict[str, float]:
    metric = metric.lower()
    if metric == "rouge":
        return evaluate.load("rouge").compute(predictions=[pred], references=[ref])
    if metric == "bleu":
        return evaluate.load("bleu").compute(predictions=[pred.split()], references=[[ref.split()]])
    if metric == "meteor":
        return evaluate.load("meteor").compute(predictions=[pred], references=[ref])
    raise ValueError(f"Unsupported metric: {metric}")

