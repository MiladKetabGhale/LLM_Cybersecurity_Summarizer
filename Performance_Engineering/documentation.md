<!-- Performance_Engineering/documentation.md -->
# Remark

The Performance Engineering component is under development currently. Summaries of each stage as well as detailed results are to be produces as planned step by step. As a result, the structuring of this component will be adjusted accordingly. 

# Performance Engineering & Optimisation

Several initial optimisations (multithreaded dataloading, M1‑specific GPU utilisation, etc.) have already been implemented — see the Performance Engineering notes in the main README.md. A deeper, end‑to‑end analysis will follow once the May roadmap items (distillation, quantisation, latency tuning, etc.) have landed.

[![Status](https://img.shields.io/badge/status-planned-yellow)](../../)  
_Last updated: **May 2025** • Target release: **v0.2**_

> **Mission:** Continuously shrink latency, memory, and disk footprint of the cybersecurity‑summariser stack while safeguarding ROUGE performance.

---

## Scope

| Area                                 | Why it matters                                   |
|--------------------------------------|--------------------------------------------------|
| **Model efficiency**                 | Faster inference & smaller deployable artefacts. |
| **Data‑pipeline throughput**         | Cuts total training time, speeds experiments.    |
| **Resource utilisation (CPU / GPU)** | Enables low‑cost deployment & laptop‑level dev.  |

---

## 🗓️ Roadmap

<details>
<summary><strong>Checklist</strong> (expand)</summary>

- [ ] Distill scratch GPT‑2 → 14 M‑param student  
- [ ] Convert distilled model to **INT8 ONNX** & benchmark  
- [ ] Latency profiling notebook (`perf_latency.ipynb`)  
- [ ] Pipeline refactor: parallelised JSONL preprocessing  
- [ ] Memory audit & gradient‑checkpointing explainer  
- [ ] README update with before/after metrics table  

</details>

### Milestone table

| Week (July 2025) | Deliverable | Owner | Notes |
|-----------------|-------------|-------|-------|
| July 6 – 10      | Distillation script (`distill.py`) | @you | teacher = full GPT‑2‑scratch |
| July 11 – 17     | ONNX INT8 quant + latency report   | @you | depends on distill.py |
| July 18 – 24     | Data‑prep refactor & benchmarks    | @you | pandas → multiprocessing |
| July 25 – 31     | Memory optimisation & final doc PR | @you | includes profiling figures |

> **Tooling**: We will rely on *PyTorch 2.3*, *onnxruntime‑gpu*, and *PyTorch Profiler*.

---
## Machine Used For Benchmarking and Performance Engineering

**Operating System**
- Ubuntu 22.04.5 LTS (Jammy Jellyfish)

**CPU**
- Model: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
- Architecture: x64, 8 physical cores, 16 threads
- Max Frequency: 4.6 GHz

**Memory**
- Total System RAM: 32GB

**GPU**
- Model: NVIDIA GeForce RTX 3060 Laptop GPU
- CUDA Version: 12.8
- VRAM: 6 GB
- Driver: 570.133.07
