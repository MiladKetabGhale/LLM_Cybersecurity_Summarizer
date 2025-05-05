import os
import json
from sklearn.model_selection import train_test_split

def split_dataset(data_path, output_dir):
    """
    Loads data from `data_path`, then creates train/val/test splits,
    and saves them into the `output_dir` as JSONL files.
    """
    # Load the data
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Split the data
    train, temp = train_test_split(data, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save splits to JSONL files
    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for entry in train:
            f.write(json.dumps(entry) + "\n")

    with open(os.path.join(output_dir, "validation.jsonl"), "w") as f:
        for entry in val:
            f.write(json.dumps(entry) + "\n")

    with open(os.path.join(output_dir, "test.jsonl"), "w") as f:
        for entry in test:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # Example usage:
    # (Adjust these paths as appropriate in your environment)
    DATA_PATH = "cyber_dataset.jsonl"
    OUTPUT_DIR = "."  # Store the splits in the same folder

    split_dataset(DATA_PATH, OUTPUT_DIR)
    print("Data split into train/validation/test files successfully!")

