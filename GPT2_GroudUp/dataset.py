"""
dataset.py

Defines a PyTorch Dataset and DataLoader for autoregressive training.
Tokenizes a large input text and splits it into overlapping chunks using a sliding window.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class DatasetWrapper(Dataset):
    """
    PyTorch Dataset that returns (input_ids, target_ids) pairs for training.
    Each input is a sequence of token IDs and each target is the same sequence shifted by one.
    """
    def __init__(self, txt, tokenizer, max_length=256, stride=128):
        self.input_ids = []
        self.target_ids = []

        # Convert the full text to token IDs
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Slide over the token sequence to create overlapping input/target chunks
        for i in range(0, len(token_ids) - max_length, stride):
            # Input chunk: current window
            input_chunk = token_ids[i : i + max_length]

            # Target chunk: input chunk shifted by one
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # Convert to PyTorch tensors
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        # Return number of samples
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return (input_ids, target_ids) tuple for a given index
        return self.input_ids[idx], self.target_ids[idx]


# Function to wrap dataset into a DataLoader

def dataloader(txt, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Create dataset from input text
    dataset = DatasetWrapper(txt, tokenizer, max_length, stride)

    # Wrap dataset in PyTorch DataLoader with batching and shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
