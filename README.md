# Cybersecurity Summarization System

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

## Overview

In cybersecurity, knowledge is vast, technical, and constantly evolving. Analysts rely on sources like MITRE ATT&CK to understand threats, tactics, and vulnerabilities—but manually processing this information is time-consuming and error-prone.

This repository presents a domain-adapted summarization system which uses GPT-2 language models to generate concise, accurate summaries of cybersecurity knowledge. By combining a custom-trained GPT-2 model with fine-tuning on a curated cybersecurity dataset, we bridge the gap between general-purpose AI models and the specialized needs of cybersecurity professionals.

The system has been rigorously benchmarked against popular zero-shot summarization baselines (including BART and Pegasus). The results demonstrate competitive performance while maintaining a lightweight model footprint—despite using considerably fewer parameters than these benchmarks.

Beyond performance, the system’s modular architecture ensures easy extensibility for integrating new models or evaluation metrics. Throughout development, we prioritized reproducibility, transparent evaluation, and hardware-aware optimizations for Apple M1 systems.

### System Components

This repo is organized into the following core modules:

| Module                       | Purpose                                                               |
|-----------------------------|-----------------------------------------------------------------------|
| `DataPreprocessing/`         | Builds a curated cybersecurity summarization dataset from MITRE ATT&CK |
| `GPT2_GroundUp/`             | Defines and trains a GPT-2 model from scratch on WikiText-103         |
| `GPT2_Finetuning/`           | Fine-tunes pretrained Hugging Face GPT-2 or custom Ground-Up GPT-2     |
| `Summary_Generation_Evaluation/` | Generates summaries from fine-tuned models; evaluates outputs          |

Each module has its own `documentation.md` with detailed usage.

## Documentation Structure

This repository provides documentation at three levels:

1. **Tier 1 – `README.md` (this file):**  
   High-level project overview, system architecture, and instructions.

2. **Tier 2 – `documentation.md` files (inside each module):**  
   Detailed documentation for each module’s purpose, design decisions, and usage.

3. **Tier 3 – Inline docstrings and code comments (inside Python scripts):**  
   Function-level documentation for developers modifying or extending the code.

A detailed system architecture diagram and description are provided in [Architecture.me](https://github.com/MiladKetabGhale/LLM_Cybersecurity_Summarizer/blob/main/Architecture.me).

## Installation & Quick-Start

## Requirements

This system requires Python 3.10+ and standard ML dependencies listed in `requirements.txt`.  
Tested on macOS (Apple Silicon M1/M2) and Intel CPUs. See [Hardware Specifications](#hardware-specifications) for details.

### Setup

1. **Clone the repository:**

   ```bash
   git clone git@github.com:MiladKetabGhale/LLM_Cybersecurity_Summarizer.git
   cd LLM_Cybersecurity-Summarizer
   ```

2. **Create a conda environment:**

   ```bash
   conda create -n cyber_summarizer python=3.10
   conda activate cyber_summarizer
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

_For Apple Silicon users_:  
Ensure `tensorflow-macos`, `tensorflow-metal`, and compatible `torch` wheels are installed. See [Apple ML Support](https://developer.apple.com/metal/tensorflow-plugin/) for details.

## Running the System

Each module in this system can be run independently or as part of the full pipeline. To execute any module, navigate to its folder and follow the steps in its corresponding `documentation.md` file.

**Example workflow:**

- Prepare dataset → `DataPreprocessing/documentation.md`
- Train GPT-2 → `GPT2_GroundUp/documentation.md`
- Fine-tune → `GPT2_Finetuning/documentation.md`
- Generate summaries → `Summary_Generation_Evaluation/documentation.md`

Each `documentation.md` provides configuration details and command-line examples.

## Custom GPT-2 (Ground-Up Model)

A core contribution of this project is a **custom GPT-2 model built from the ground up** and trained on **WikiText-103** (103 million tokens). We later fine-tuned it on the curated cybersecurity dataset (amounting to 125K tokens) created during preprocessing.  
The fine-tuned model is benchmarked for direct performance comparison with Hugging Face GPT-2.

### Model Architecture

| Parameter            | Value            |
|---------------------|-----------------|
| Number of layers     | 12              |
| Attention heads      | 4               |
| Embedding dimension  | 256             |
| Context length       | 256             |
| Dropout              | 0.0             |
| Total parameters     | ~22.3 million   |

## Dataset Preparation & Validation

We curated a high-quality **cybersecurity summarization dataset** through a multi-stage pipeline:

1. **Corpus extraction:**  
   - Scraped **435 entries** (amounting to 125K tokens) from [MITRE ATT&CK](https://attack.mitre.org) using a custom extraction script.
2. **Gold summary generation:**  
   - Generated reference (“gold”) summaries for each entry using a **locally deployed Distilled DeepSeek R1 (LLaMA 7B)** model via tailored prompt engineering.
3. **Validation process:**
   - **Statistical validation:**  
     Compared the generated summaries to GPT-4o outputs on a random 10% sample using **t-test** and **Kolmogorov–Smirnov test**.  
     → *No statistically significant difference in ROUGE metrics observed.*
   - **Manual validation:**  
     Conducted **human-in-the-loop evaluation** on the same 10% sample for coherence, completeness, and factual accuracy.  
     → *Confirmed acceptable quality for training use.*

This dataset was subsequently used to fine-tune both the **custom ground-up GPT-2 model** and the **pretrained GPT-2 model**.

## Evaluations and Performance Optimizations

Below we report benchmark results, describe the engineering optimizations that enabled them, and specify the hardware context to aid reproducibility. ***Please note that 
the benchmarks are soon going to be updated based on results obtained with a different system using a single NVIDIA GPU (RTX3060)***.

### Benchmark Results

We benchmarked the models on the curated cybersecurity dataset:

| Model                    | Parameters       | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------------------|-----------------|----------|----------|----------|
| GPT-2 (LoRA fine-tuned)  | < 1% trained      | 0.1811   | 0.0257   | 0.1284   |
| GPT-2 (fully fine-tuned) | 124M trained    | 0.2326   | 0.0460   | 0.1598   |
| BART (zero-shot)         | 139M total      | 0.3271   | 0.1059   | 0.2144   |
| Pegasus (zero-shot)      | 568M total      | 0.1914   | 0.0528   | 0.1402   |

Fine-tuned GPT-2 significantly reduced the gap to BART zero-shot.  
LoRA fine-tuning achieved ~80% of full fine-tuned ROUGE-1 performance with <1% parameters fine tuned.

#### Inference Time Comparison

To complement evaluation metrics (ROUGE), we benchmarked inference time, input/output lengths, and token efficiency for each model:

| Metric             | BART-base        | Pegasus         | GPT-2 (Fully Fine-tuned) | GPT-2 (LoRA Fine-tuned) |
|--------------------|------------------|-----------------|--------------------------|-------------------------|
| Inference time     | 3357.71 ms         | 3971.57 ms      | 2356.73 ms               | ~2356.73 ms             |
| Input tokens       | 211.75 tokens      | 204.87 tokens   | 252.88 tokens            | 252.88 tokens           |
| Output tokens      | 98.67 tokens       | 23.28 tokens    | 100.00 tokens            | 100.00 tokens           |
| Time per token     | 34.13 ms/token     | 181.56 ms/token | 23.57 ms/token           | ~23.57 ms/token         |

> **Note:** The LoRA-based GPT-2 model shows no difference in two decimal points in inference time compared to full fine-tuning. This is *expected behavior* because LoRA modifies only a small subset of trainable parameters, and—when merged into the base model—it does **not introduce runtime overhead**.

### Performance Engineering

We optimized training on Apple M1 hardware:

- Added **multithreaded data loading** to saturate GPU → increased GPU utilization from **30% → 100%**
- CPU utilization dropped from **700% → 70% per trial** after moving preprocessing off CPU
- Achieved **~3× faster per-trial runtime**

These improvements enabled efficient experimentation even on constrained hardware.

### Hardware Specifications

This project was developed and tested on the following system:

- **Model Name:** MacBook Pro
- **Model Identifier:** MacBookPro18,1
- **Chip:** Apple M1 Pro
- **CPU:** 10-core CPU (8 performance cores + 2 efficiency cores)
- **GPU:** 16-core integrated GPU (Apple M1 Pro, Metal GPUFamily Apple 7)
- **Memory:** 16 GB unified memory
- **Vendor:** Apple (0x106b)
- **OS Loader Version:** 11881.81.4

_Note: Serial number, hardware UUID, and provisioning UDID have been excluded for privacy._

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
You are free to use, modify, and distribute the code, provided proper attribution is given.  
The software is provided "as is," without warranties or guarantees of any kind.

## Acknowledgments

I benefited from a number of resources throughout the building process of the project which I like to acknowledge and thank. 

- [MITRE ATT&CK](https://attack.mitre.org)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
- [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [OpenAI GPT-4o](https://platform.openai.com/docs/models/gpt-4o)

- See [Resources.md](Resources.md) for a full list of references and learning materials used in this project.
