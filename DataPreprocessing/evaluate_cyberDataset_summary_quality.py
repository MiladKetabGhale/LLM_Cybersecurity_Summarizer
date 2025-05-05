from transformers import GPT2TokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# Your dataset
data = [
    {
        "source": "...",  # fill with your full dataset
        "summary": "..."
    },
    # ... more entries
]

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Tokenize
tokenized_refs = [tokenizer.tokenize(d["source"].lower()) for d in data]
tokenized_preds = [tokenizer.tokenize(d["summary"].lower()) for d in data]

# ROUGE-1
def compute_rouge1(ref_tokens, pred_tokens):
    ref_counts = {}
    pred_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = sum(min(ref_counts.get(tok, 0), pred_counts.get(tok, 0)) for tok in pred_counts)
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    f1 = 2 * precision * recall / max((precision + recall), 1e-8)
    return precision, recall, f1

# Evaluation
rouge_precisions, rouge_recalls, rouge_f1s, bleu_scores = [], [], [], []
for ref, pred in zip(tokenized_refs, tokenized_preds):
    p, r, f1 = compute_rouge1(ref, pred)
    rouge_precisions.append(p)
    rouge_recalls.append(r)
    rouge_f1s.append(f1)
    bleu = sentence_bleu([ref], pred, smoothing_function=SmoothingFunction().method1)
    bleu_scores.append(bleu)

print({
    "ROUGE-1 Precision": np.mean(rouge_precisions),
    "ROUGE-1 Recall": np.mean(rouge_recalls),
    "ROUGE-1 F1": np.mean(rouge_f1s),
    "BLEU": np.mean(bleu_scores)
})

