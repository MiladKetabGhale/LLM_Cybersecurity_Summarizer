"""
generate.py

Defines generation functions for autoregressive GPT-style decoding.
Includes greedy decoding and top-k sampling with temperature.
"""

import torch
import torch.nn.functional as F

def generate_summary(model, prompt_ids, max_tokens=50, device='cpu', end_token_id=50256):
    """
    Greedy decoding: generates tokens one-by-one by selecting the highest probability token.
    """
    model.eval()
    model.to(device)

    generated = prompt_ids.clone().to(device)
    max_context_length = model.position_emb.num_embeddings

    with torch.no_grad():
        for _ in range(max_tokens):
            # Clip to context length if needed
            if generated.size(1) > max_context_length:
                generated = generated[:, -max_context_length:]

            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == end_token_id).all():
                break

    return generated


def generate_summary_topk_temperature(
    model,
    prompt_ids: torch.Tensor,
    max_tokens: int = 50,
    k: int = 5,
    temperature: float = 1.0,
    device: str = "cpu",
    end_token_id: int = 50256):
    """
    Top-k sampling with temperature for more diverse generation.
    """
    model.eval()
    model.to(device)

    generated = prompt_ids.clone().to(device)
    max_context_length = model.position_emb.num_embeddings

    with torch.no_grad():
        for _ in range(max_tokens):
            # Clip to context length if needed
            if generated.size(1) > max_context_length:
                generated = generated[:, -max_context_length:]

            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            topk_logits, topk_indices = torch.topk(next_token_logits, k)
            probs = F.softmax(topk_logits, dim=-1)
            next_token = topk_indices.gather(1, torch.multinomial(probs, num_samples=1))
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == end_token_id).all():
                break

    return generated


def generate_summary_on_input(model, tokenizer, input_text, max_len=50, k=5, temperature=1.0):
    """
    Wrapper function that accepts raw input text, tokenizes it,
    performs top-k sampling with temperature, and returns the generated summary.
    """
    prompt = f"Summarize: {input_text.strip()} TL;DR:"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = generate_summary_topk_temperature(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=max_len,
        k=k,
        temperature=temperature,
        device=next(model.parameters()).device,
        end_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

