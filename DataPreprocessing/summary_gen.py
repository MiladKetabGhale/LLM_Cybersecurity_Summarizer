import requests
import json
import os

def summarize_with_lm_studio(input_text: str) -> str:
    # Build the prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes cybersecurity definitions clearly and concisely."
        },
        {
            "role": "user",
            "content": f"Summarize the following cybersecurity concept in at most three lines for a technical audience:\n\n{input_text}"
        }
    ]

    payload = {
        "model": "local-model",  # LM Studio doesn't use this field, placeholder is fine
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 512
    }

    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content.strip()
    else:
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text}")

def save_summary_pair(original_text: str, summary_text: str, output_file: str = "cyber_dataset.jsonl"):
    # Ensure consistent format for fine-tuning (e.g., HuggingFace style)
    example = {
        "source": original_text.strip(),
        "summary": summary_text.strip()
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

# Example input (can later be a loop over scraped MITRE pages)
example_text = """
Man-in-the-middle (MitM) attacks are a class of cyberattacks where the adversary secretly intercepts and possibly alters the communication between two parties. This technique can be used for credential theft, session hijacking, or injecting malicious content, and it often exploits weaknesses in network protocols, SSL/TLS certificates, or user behavior.
"""

# Generate summary
summary = summarize_with_lm_studio(example_text)
print("Summary:\n", summary)

# Save the (input, summary) pair
save_summary_pair(example_text, summary)

