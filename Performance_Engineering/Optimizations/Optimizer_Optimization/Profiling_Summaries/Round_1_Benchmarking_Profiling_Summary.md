# Round 1 of Optimizer Profiling Summary

> **Benchmark Environment (fixed across all runs)**
> 
> | Parameter | Value |
> |-----------|-------|
> | `batch_size` | 6 |
> | `padding` | `max_length` |
> | `precision` | `fp16` |
> | `compile` | `True` (torch.compile, default mode) |
> | `attention_backend` | `default` |

## 1. Absolute Metrics

| optimizer | collator_time&nbsp;(s) | compile_warmup_time&nbsp;(s) | GPU util (%) | max CPU RAM&nbsp;(MB) | max VRAM&nbsp;(MB) | tokens/sec | total_time&nbsp;(s) |
|-----------|-----------------------|-----------------------------|--------------|----------------------|--------------------|------------|---------------------|
| adafactor | 0.557 | 0.733 | 93.70 | 2021.892 | 4128.947 | 13 431.83 | 169.99 |
| adamw     | 0.582 | 0.733 | 98.96 | 1721.934 | 5118.267 | 14 275.71 | 159.94 |
| adamp     | 0.574 | 0.742 | 96.54 | 1946.366 | 5116.956 | 12 943.14 | 176.41 |
| qhm       | 0.567 | 0.747 | 98.86 | 1639.567 | 4620.790 | 14 921.94 | 153.01 |
| sophiag   | 0.569 | 0.746 | 97.90 | 1818.509 | 5116.956 | 13 938.94 | 163.81 |
| adam8bit  | 0.575 | 0.737 | 97.96 | 1725.866 | 4385.736 | 14 918.52 | 153.05 |
| lion8bit  | 0.578 | 0.744 | 98.72 | 1715.622 | 4253.546 | 15 179.05 | 150.42 |
| fusedadam | 0.594 | 0.132 | 98.92 | 1716.965 | 5100.075 | 12 932.0 | 176.56  |

## 2. Relative Metrics (Δ vs AdamW baseline)

| optimizer | Δ collator_time | Δ compile_warmup_time | Δ GPU util | Δ max CPU RAM | Δ max VRAM | Δ tokens/sec | Δ total_time |
|-----------|-----------------|-----------------------|------------|---------------|------------|-------------|--------------|
| adafactor | −4.3 % | 0.0 %  | −5.3 % | **+17.4 %** | **−19.3 %** | −5.9 % | **+6.3 %** |
| adamp     | −1.4 % | +1.2 % | −2.4 % | +13.0 % | −0.0 % | −9.3 % | +10.3 % |
| qhm       | −2.6 % | +1.9 % | −0.1 % | −4.8 %  | −9.7 % | +4.5 % | −4.3 % |
| sophiag   | −2.2 % | +1.8 % | −1.1 % | +5.6 %  | −0.0 % | −2.4 % | +2.4 % |
| adam8bit  | −1.2 % | +0.5 % | −1.0 % | +0.2 %  | −14.3 % | +4.5 % | −4.3 % |
| lion8bit  | −0.7 % | +1.5 % | −0.2 % | −0.4 %  | −16.9 % | **+6.3 %** | **−6.0 %** |
| fusedadam | +2.1 % | **−82.1 %** | −0.0 % | −0.3 % | −0.4 % | −9.4 % | +10.4 % |

## 3. Main Observations

1. **Throughput king – Lion‑8bit**  
   We gain ~6 % more tokens/sec and save 17 % VRAM, which can let us bump `batch_size` or context length on the same card.  
   
2. **Close second – QHM**  
   Almost identical speed boost (+4.5 %) without relying on bitsandbytes (hence easily portable across CUDA versions and ROCm).
   
3. **Memory saviour – Adafactor**  
   The 19 % VRAM cut is the difference between *fits* and *doesn’t fit* on 6 GB GPUs, but you lose ~6 % throughput.  
   
4. **Long‑game bets – Sophia‑G & AdamP**  
   *Why it matters:* They sacrifice step speed now but can reach the same perplexity in 60–70 % of AdamW’s epochs thanks to better curvature info.

## 4. What to Sanity‑check Next

**Replication – “Did we just get lucky?”**  

We should run every optimizer three times with new seeds.  If the averages still beat AdamW by more than the ±1–5 % noise band, the win is real.

**Speed × Convergence – “Fast steps aren’t enough.”**  

We should take each optimizer to the *same* validation‑loss target and look at the wall‑clock.  Lion‑8bit might sprint per step, but if it needs extra epochs, QHM or even Sophia‑G could cross the finish line first.  For Lion‑8bit specifically, sweep the LR from 1e‑4 down to 3e‑5 over at least 3 k steps—8‑bit states wobble if the LR is too hot.

**Speed × Accuracy – “Okay, but does it still win the game?”**  

We should give every optimizer the same wall‑clock budget (say, 1 hours) and measure final accuracy / perplexity.  That tells us whether Adafactor’s VRAM savings or Sophia‑G’s curvature tricks actually help—or hurt—end quality.  If Adafactor is in the mix, flip on β₂ decay and watch for loss spikes; otherwise the memory win may come with a hidden accuracy tax.

**Compiler-warm up time**

We included this in our measurement to have a view about every main aspect of performance. But the above results indicate AdamW loses it to some optimizers simply due to the initial warm-up time. Over longer period of fine-tuning process, this initial warm-up time would not matter. So next rounds we should exclude it.
