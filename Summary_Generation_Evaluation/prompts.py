import json
from pathlib import Path
from typing import Union

def gpt2_prompt(source: str) -> str:
    return (
        "<|summarize|> Document: "
        f"{source}\n\n"
        "### Task: In two concise sentence (≤ 75 tokens) summarise the document for a cybersecurity analyst"
        "for a cybersecurity analyst.\n\n"
        "### Summary:"
    )

