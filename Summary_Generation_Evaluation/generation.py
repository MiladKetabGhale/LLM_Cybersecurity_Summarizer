from transformers import StoppingCriteria, StoppingCriteriaList, GPT2Tokenizer

# ────────────────────────── GENERATION UTILS ────────────────────────────────

class StopAtPeriod(StoppingCriteria):
    """Stop GPT‑2 generation after first period (.) beyond *min_tokens*."""

    def __init__(self, tokenizer: GPT2Tokenizer, min_tokens: int = 6):
        self.tok_period = tokenizer.encode(".")[0]
        self.min_tokens = min_tokens

    def __call__(self, input_ids, scores, **kwargs):
        return (
            input_ids.shape[-1] >= self.min_tokens and input_ids[0, -1] == self.tok_period
        )

# ---------- extraction helpers ---------------------------------------------

def extract_summary(decoded: str, prompt: str) -> str:
    if "### Summary:" in decoded:
        return decoded.split("### Summary:", 1)[1].strip()
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    return decoded.strip()


# ---------- generators ------------------------------------------------------

def generate_decoder(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    stopping = StoppingCriteriaList([StopAtPeriod(tokenizer)])
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
        stopping_criteria=stopping,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return extract_summary(decoded, prompt)


def generate_seq2seq(
    model,
    tokenizer,
    source: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(source, return_tensors="pt", truncation=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

# generation.py  (append near the bottom)

def build_few_shot_prompt(examples, new_source):
    """
    Few-shot prompt format:
    Source: ...
    Summary: ...

    Source: ...
    Summary: ...

    Source: <new_source>
    Summary:
    """
    parts = [
        f"Source: {ex['source']}\nSummary: {ex['summary']}\n"
        for ex in examples
    ]
    parts.append(f"Source: {new_source}\nSummary:")
    return "\n".join(parts)


def generate_few_shot(model, tokenizer, source, examples, max_new_tokens):
    """
    Wrapper that builds the prompt and calls the decoder/seq2seq path
    transparently.
    """
    prompt = build_few_shot_prompt(examples, source)

    if hasattr(model.config, "is_decoder_only") or tokenizer.__class__.__name__.startswith("GPT"):
        return generate_decoder(model, tokenizer, prompt, max_new_tokens)
    return generate_seq2seq(model, tokenizer, prompt, max_new_tokens)
