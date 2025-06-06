# DataPreprocessing Module Documentation

## Overview

The **DataPreprocessing** module handles the construction of a clean, structured dataset for cybersecurity text summarization.  
It ingests raw data from the MITRE ATT&CK corpus, extracts relevant descriptions, generates preliminary summaries, and splits the dataset into training, validation, and test sets for downstream model training and evaluation.

This module provides a reproducible data preparation pipeline for training summarization models on curated cybersecurity data.

---

## Entry Point

The main orchestration script for this module is:

```bash
python DataPreprocessing/mitre_corpus_extrator.py
python DataPreprocessing/build_cybersecurity_dataset.py
```

This script sequentially runs extraction, summarization, and splitting steps to produce the following output files in this folder:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

Alternatively, each processing step can be executed independently via its dedicated script.

---

## File Descriptions

| File                                | Purpose                                                                                                                                       |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `mitre_corpus_extractor.py`        | Extracts textual descriptions from the raw MITRE ATT&CK JSON source files and writes them into a simplified intermediate JSONL file (`mitre_descriptions.jsonl`). <br> → **Input**: MITRE JSON <br> → **Output**: `mitre_descriptions.jsonl` |
| `summary_gen.py`                    | Uses prompt templates or heuristics to auto-generate initial summary candidates from the extracted descriptions, producing `cyber_dataset.jsonl`. <br> → **Input**: `mitre_descriptions.jsonl` <br> → **Output**: `cyber_dataset.jsonl` |
| `build_cybersecurity_dataset.py`    | Orchestrates the end-to-end data preparation pipeline by calling the extractor, summarizer, and splitter in sequence to build the dataset. Acts as a wrapper combining all steps. |
| `data_split.py`                     | Splits the `cyber_dataset.jsonl` into training, validation, and testing splits, saving them as `train.jsonl`, `validation.jsonl`, and `test.jsonl`. |
| `evaluate_cyberDataset_summary_quality.py` | Computes quality metrics (including ROUGE) over the generated dataset summaries to assess their summarization fidelity before feeding them into model training. |

---

## Execution Flow

A typical full pipeline execution follows this sequence:

1. **Extract MITRE descriptions:**

   ```bash
   python DataPreprocessing/mitre_corpus_extractor.py
   # → writes mitre_descriptions.jsonl
   ```

2. **Generate summaries:**

   ```bash
   python DataPreprocessing/summary_gen.py
   # → writes cyber_dataset.jsonl
   ```

3. **Split into train/validation/test:**

   ```bash
   python DataPreprocessing/data_split.py
   # → writes train.jsonl, validation.jsonl, test.jsonl
   ```

4. **Evaluate dataset quality (optional diagnostic):**

   ```bash
   python DataPreprocessing/evaluate_cyberDataset_summary_quality.py
   ```

Alternatively, the entire pipeline can be executed via the combined entry point:

```bash
python DataPreprocessing/build_cybersecurity_dataset.py
```

---

## Output Files

This module produces the following key outputs:

- `mitre_descriptions.jsonl`: extracted raw descriptions
- `cyber_dataset.jsonl`: input–summary pairs
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: dataset splits for training and evaluation

---

## Dependencies

- Requires a MITRE ATT&CK JSON source file as input (path hardcoded or configured inside `mitre_corpus_extractor.py`)
- Written in **Python 3.10+**
- External libraries used:
  - `json`
  - `pandas`
  - `sklearn.model_selection` (for splitting)
  - `rouge_score` (for evaluation)

No additional configuration files are required; paths and constants are set within the scripts.
