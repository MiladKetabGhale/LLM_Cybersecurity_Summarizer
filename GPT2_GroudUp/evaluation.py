"""
evaluation.py

Evaluates a summarization model using ROUGE scores based on predicted vs. reference summaries.
Includes top-k sampling with temperature to generate outputs.
"""

import torch
from rouge_score import rouge_scorer
from generate import generate_summary_topk_temperature


def evaluate_model(model, tokenizer, dataset, max_len, k=5, temperature=1.0, limit=100, device="cpu"):
    """
    Evaluate the model on a dataset using ROUGE metrics.

    Args:
        model: Trained transformer model.
        tokenizer: Tokenizer for encoding/decoding text.
        dataset (list[dict]): Each item should have 'dialogue' and 'summary'.
        max_len (int): Maximum number of tokens to generate.
        k (int): Top-k sampling size.
        temperature (float): Sampling temperature.
        limit (int): Max number of samples to evaluate.

    Returns:
        None
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    total_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    n = 0

    for example in dataset:
        dialogue = example.get("source", "").strip().replace("\n", " ")
        reference = example.get("summary", "").strip()

        if not dialogue or not reference:
            continue

        # Construct the prompt
        prompt = f"Summarize: {dialogue} TL;DR:"
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        # Generate summary using top-k sampling with temperature
        generated_ids = generate_summary_topk_temperature(
            model=model,
            prompt_ids=prompt_ids,
            max_tokens=max_len,
            k=k,
            temperature=temperature,
            device="cpu",
            end_token_id=tokenizer.encode("<|endoftext|>")[0]
        )

        # Decode and clean summary output
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        generated_summary = generated_text.split("TL;DR:")[-1].strip().replace("<|endoftext|>", "")

        # Compute ROUGE scores
        scores = scorer.score(reference, generated_summary)
        for key in total_scores:
            total_scores[key] += scores[key].fmeasure
        n += 1

        if n >= limit:
            break

    if n == 0:
        print("[WARNING] No valid evaluation examples found.")
        return

    print(f"\n📊 Evaluation on {n} validation samples:")
    for key in total_scores:
        print(f"{key.upper():<8}: {total_scores[key] / n:.4f}")

