import json
import numpy as np
import onnxruntime
from transformers import GPT2Tokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

session = onnxruntime.InferenceSession("gpt2_finetuned_int8.onnx", providers=["CPUExecutionProvider"])
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def generate_summary(text, max_new_tokens=100):
    inputs = tokenizer(text, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"]
    generated = input_ids.copy()

    for _ in range(max_new_tokens):
        attention_mask = np.ones_like(generated)
        position_ids = np.arange(generated.shape[1])[None, :]  # [1, seq_len]

        ort_inputs = {
            "input_ids": generated,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

        outputs = session.run(None, ort_inputs)
        next_token_logits = outputs[0][:, -1, :]
        next_token = np.argmax(next_token_logits, axis=-1)[..., None]

        generated = np.concatenate([generated, next_token], axis=1)

        if next_token[0][0] == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def evaluate_onnx_model(input_path):
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    with open(input_path, "r") as f:
        for line in tqdm(f, desc="Evaluating"):
            item = json.loads(line)
            source_text = item["source"]
            reference_summary = item["summary"]

            generated_summary = generate_summary(source_text)

            scores = scorer.score(reference_summary, generated_summary)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

    # Aggregate scores
    avg_scores = {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }

    return avg_scores


if __name__ == "__main__":
    input_file = "INPUT_DATA_PATH"  # <- each line is {"source": ..., "summary": ...}
    scores = evaluate_onnx_model(input_file)
    print("\nAverage ROUGE scores:")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

