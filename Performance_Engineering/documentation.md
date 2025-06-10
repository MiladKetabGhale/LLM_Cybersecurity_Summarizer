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

## 🗓️ May 2025 Roadmap

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

| Week (May 2025) | Deliverable | Owner | Notes |
|-----------------|-------------|-------|-------|
| May 6 – 10      | Distillation script (`distill.py`) | @you | teacher = full GPT‑2‑scratch |
| May 11 – 17     | ONNX INT8 quant + latency report   | @you | depends on distill.py |
| May 18 – 24     | Data‑prep refactor & benchmarks    | @you | pandas → multiprocessing |
| May 25 – 31     | Memory optimisation & final doc PR | @you | includes profiling figures |

> **Tooling**: We will rely on *PyTorch 2.3*, *onnxruntime‑gpu*, and *PyTorch Profiler*.

---

## Future sections (will appear as PRs land)

```text
Performance_Engineering/
├── distill.py                  # knowledge‑distillation driver
├── quantize_onnx.py            # export + INT8 calibration
├── perf_latency.ipynb          # latency profiling notebook
└── README_perf_results.md      # side‑by‑side before/after metrics
