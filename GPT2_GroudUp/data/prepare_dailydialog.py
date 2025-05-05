from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch

def load_and_prepare_dataset(config):
    tokenizer = GPT2Tokenizer.from_pretrained(config["tokenizer"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(config["data"]["dataset_name"])
    train_data = dataset["train"]
    val_data = dataset["validation"] if "validation" in dataset else dataset["test"]

    input_ids = []

    for example in train_data:
        if "dialog" in example:
            # Join list of utterances into a single dialogue string
            dialog = " ".join(example["dialog"]).strip()
            text = f"{dialog} {tokenizer.eos_token}"
            tokens = tokenizer.encode(text)
            input_ids.extend(tokens)
        else:
            raise ValueError("Expected 'dialog' field in DailyDialog dataset")

    input_ids = torch.tensor(input_ids, dtype=torch.long)

    max_length = config["data"]["max_length"]
    stride = config["data"]["stride"]

    chunks = []
    for i in range(0, len(input_ids) - max_length + 1, stride):
        chunk = input_ids[i : i + max_length]
        chunks.append(chunk)

    if len(input_ids) % stride != 0:
        chunk = input_ids[-max_length:]
        chunks.append(chunk)

    train_tensors = torch.stack(chunks)

    return train_tensors, val_data

