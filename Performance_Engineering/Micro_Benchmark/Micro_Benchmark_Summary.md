# Microbenchmarking GPT-2: Runtime Dissection and Bottleneck Diagnosis

Following the initial baseline benchmarking—which indicated that a batch size of 5 with Torch compilation enabled gave a strong balance between throughput and latency on our 6GB RTX 3060—we now shift focus from macro-level trends to micro-level performance composition.

This phase dissects the training loop into distinct runtime segments, enabling us to measure how much time is truly spent in each component (e.g., forward pass, optimizer step, CUDA sync). By understanding the relative and absolute costs of each part of the pipeline, we gain insight into **what truly dominates training time** and whether the observed performance improvements are **compute-bound, memory-bound, or I/O-bound**.

The motivation for this step is twofold:
1. **Identify bottlenecks** that aren’t visible from aggregate metrics like tokens/sec alone.
2. **Guide optimization efforts** toward the components most likely to yield significant gains.

We test three configurations:
- Run 1: Baseline (batch size 5, default attention)
- Run 2: Reduced overhead (same batch size, profiling overhead minimized)
- Run 3: Batch size 6 (pushing GPU memory to the practical limit)

Each run is analyzed for **component-wise time contributions**, **peak memory usage**, and **throughput**. This lets us distinguish between overhead-dominated and compute-dominated phases, and helps validate whether increasing batch size leads to proportionally better performance—or just inflated memory pressure with diminishing returns.

---
## Relative Time Breakdown (% of Total Runtime)

| Component               | Run 1(bs=5, default) | Run 2(bs=5, reduce-overhead) | Run 3(bs=6, default)    |
|-------------------------|---------------------------|-------------------------|-------------------------|
| Offline Tokenisation    | 0.096%                    | 0.097%                  | 0.077%                  |
| Collator CPU Time       | 0.330%                    | 0.328%                  | 0.319%                  |
| Dataloader Wait         | 0.367%                    | 0.366%                  | 0.349%                  |
| Transfer to Device      | 0.025%                    | 0.025%                  | 0.020%                  |
| Forward Pass            | 11.20%                    | 12.96%                  | 19.28%                  |
| Backward Pass           | 5.18%                     | 5.72%                   | 5.25%                   |
| Optimiser Step          | 66.26%                    | 66.83%                  | 65.11%                  |
| CUDA Sync               | 10.53%                    | 10.22%                  | 8.35%                   |
| Unaccounted/Other       | 5.97%                     | 2.46%                   | 1.28%                   |
| **Total Measured Time** | **145.10 s**              | **150.75 s**            | **183.72 s**            |

The optimizer step dominates total runtime (~65–67%) across all configurations, confirming it as the primary bottleneck. The forward pass time increases sharply when batch size is raised from 5 to 6 (11.2% → 19.3%). This suggests compute scaling is sublinear and memory contention may be rising.

---
## Benchmarking Summary (Baseline bs=5, reduce-overhead)
Relative time percentages help reveal bottlenecks, but absolute durations (shown in the Benchmarking Summary table) give a more actionable view of where time is truly being spent.

| Metric               | Run 1(bs=5, default) | Baseline(bs=5, reduce-overhead) | Run 3(bs=6, default) |
|----------------------|---------------------------|----------------------------|---------------------------|
| Tokens/sec           | 3796.99 (0.00%)           | 3796.99                    | 4566.53 (+20.26%)         |
| Offline Tokenisation | 0.1409 s (−3.00%)         | 0.1452 s                   | 0.1410 s (−2.89%)         |
| Collator CPU Time    | 0.4827 s (−1.45%)         | 0.4898 s                   | 0.5867 s (+19.80%)        |
| Dataloader Wait      | 0.5367 s (−1.86%)         | 0.5469 s                   | 0.6429 s (+17.56%)        |
| Transfer to Device   | 0.0359 s (−4.27%)         | 0.0375 s                   | 0.0362 s (−3.47%)         |
| Forward Pass         | 16.3507 s (−16.06%)       | 19.4777 s                  | 35.3227 s (+81.30%)       |
| Backward Pass        | 7.5569 s (−12.12%)        | 8.5990 s                   | 9.6103 s (+11.76%)        |
| Optimizer Step       | 96.6426 s (−3.84%)        | 100.5010 s                 | 119.2858 s (+18.69%)      |
| CUDA Sync            | 15.3526 s (−0.02%)        | 15.3558 s                  | 15.2965 s (−0.39%)        |
| Peak GPU VRAM        | 3951.9 MB (+0.59%)        | 3928.6 MB                  | 5096.0 MB (+29.69%)       |
| Peak CPU RAM         | 1753.6 MB (−48.89%)       | 3429.7 MB                  | 1804.4 MB (−47.40%)       |
| GPU Utilization      | 98.68% (+0.24%)           | 98.44%                     | 97.92% (−0.53%)           |
| Compile Warmup Time  | 0.7159 s (−1.54%)         | 0.7272 s                   | 0.7215 s (−0.78%)         |

Increasing batch size to 6 boosts throughput by ~20%, but also leads to a significant rise in forward pass time (+81%) and GPU VRAM usage (+30%), indicating diminishing efficiency. The “reduce-overhead” config (Run 2) shows marginal improvements in most subcomponents, but not enough to shift performance meaningfully.

---
## Optimization Recommendations

### Replace AdamW with a More Efficient Optimizer

The optimizer step currently takes up most of your runtime. Swapping AdamW with a more efficient alternative can lead to significant performance gains:

- **Adam8bit**: Offers the best speed and memory savings on consumer GPUs. Uses 8-bit optimizer states and fused GPU kernels.
- **Adafactor**: Extremely memory-efficient, suitable for long sequences. Used in T5 models.
- **Lion**: A newer optimizer with a simple, fast update rule. Lightweight and performs well on transformer models.

### Split Optimizer Timing into Subcomponents

Right now, all optimizer-related operations are grouped under one timer. To understand where the real cost is, break this down into:

- Time spent on `opt.step()` (parameter updates)
- Time spent on `scaler.update()` (AMP scaling)
- Time spent on `opt.zero_grad()` (gradient reset)

This gives clarity on which specific operation is the bottleneck.

### Enable Gradient Checkpointing

This reduces VRAM usage by recomputing intermediate activations during the backward pass instead of storing them during the forward pass. On a 6GB GPU, this frees up space and allows for larger or more stable configurations.

### Use Gradient Accumulation

Memory is limiting physical batch size. We should try to accumulate gradients over multiple smaller steps to simulate a larger logical batch size. This is useful for improving training stability and optimizer behavior without needing additional memory.
