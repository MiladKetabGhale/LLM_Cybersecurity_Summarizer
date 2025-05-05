import json
import requests

INPUT_FILE = "mitre_descriptions.jsonl"
OUTPUT_FILE = "cyber_dataset.jsonl"

def summarize_with_lm_studio(input_text: str) -> str:
    """ Summarize a cybersecurity concept using a local LM Studio API server.

    This function sends the input_text to the locally hosted language model Distilled R1 (into Llama, 7B parameters) 
    exposed via LM Studio's OpenAI-compatible HTTP API 
    running at http://localhost:1234.

    Args:
        input_text (str): The input description to summarize.
    Returns:
        str: The generated summary text.
    """

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
        "model": "local-model",  # Placeholder, ignored by LM Studio
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
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text}")

def save_summary_pair(original_text: str, summary_text: str):
    """Save a (source, summary) pair to OUTPUT_FILE in JSONL format.

    Each saved entry represents an original MITRE ATT&CK description and its
    generated summary, written as a JSON object per line.
    
    Args:
        original_text (str): The original input text.
        summary_text (str): The generated summary.
    """

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        json.dump({
            "source": original_text.strip(),
            "summary": summary_text.strip()
        }, f, ensure_ascii=False)
        f.write("\n")

def main():
    """Iterate over input JSONL, summarize each description, and save outputs.

    Reads MITRE ATT&CK descriptions from INPUT_FILE, generates summaries using
    a locally hosted distilled model served via LM Studio, and appends results
    to OUTPUT_FILE in JSONL format.
    """

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
            original_text = entry.get("description", "").strip()
            if not original_text:
                print(f"[{idx}] Skipped: Empty description.")
                continue

            print(f"[{idx}] Summarizing...")
            summary = summarize_with_lm_studio(original_text)
            save_summary_pair(original_text, summary)
            print(f"[{idx}] Saved.")

        except Exception as e:
            print(f"[{idx}] Error: {e}")

if __name__ == "__main__":
    main()

