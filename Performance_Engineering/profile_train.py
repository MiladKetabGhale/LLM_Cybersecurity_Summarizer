import time  #Added profiling support related
import torch
import os
import json
from tqdm import tqdm
from torch import nn
from transformers import GPT2Tokenizer
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler  #Added profiling support related #o4 fixes added
from torch.profiler import schedule as profiler_schedule
from torch.utils.tensorboard import SummaryWriter

# ── constants ────────────────────────────────────────────────────────────
_tok = GPT2Tokenizer.from_pretrained("gpt2")
_tok.pad_token = _tok.eos_token
PAD_ID = _tok.pad_token_id
CKPT_DIR = "GroundUp_ModelTraining_Outcome"

# ── helper: validation loss ──────────────────────────────────────────────
def evaluate(
    model,
    val_loader,
    device,
    loss_fn,
    subset_ratio: float = 1.0,
):
    """Return average validation loss over *subset_ratio* of val_loader."""
    model.eval()
    running, seen = 0.0, 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if subset_ratio < 1.0 and (idx / len(val_loader)) >= subset_ratio:
                break
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            running += loss.item()
            seen += 1
    return running / max(seen, 1)

# ── training loop with epoch- & step-level early stopping ────────────────
def train_summarizer(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    device="cpu",
    num_epochs: int = 20,
    patience: int = 4,              # epoch-level patience
    val_every_steps: int = 8000,   # quick-validation frequency
    step_patience: int = 4,         # patience for quick-validation
    val_subset: float = 0.2,        # fraction of val set for quick-check
    profile_top_n: int = 10         # number of top ops to summarize
):
    """
    Two-tier early-stopping:

    • Quick validation every *val_every_steps* steps inside an epoch.
      Abort the rest of the epoch after *step_patience* consecutive
      non-improving quick checks.

    • Full validation at the end of each epoch.
      Stop training after *patience* consecutive non-improving epochs.
    """
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    #Added profiling support related: define timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")  #Added profiling support related
    
    os.makedirs(f'{CKPT_DIR}/Profilerlogs/run_{timestamp}', exist_ok=True)
    writer = SummaryWriter(f'{CKPT_DIR}/Profilerlogs/run_{timestamp}')

    #Added profiling support related: set up PyTorch CPU profiler
    tb_handler = tensorboard_trace_handler(f'{CKPT_DIR}/Profilerlogs/run_{timestamp}')
    with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tb_handler,
            schedule=profiler_schedule(
                wait=8000,    # skip the first 7999 steps
                warmup=0,      # no warmup period
                active=1,      # profile exactly 1 step
                repeat=8       # then repeat 8 more times ⇒ 9 profiles total
            ),  # profiles at steps 800, 1600, …, up to 7200
        ) as prof:

        train_curve, val_curve = [], []
        best_across_epochs = float("inf")
        epoch_no_improve = 0
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()  #Added profiling support related
            model.train()
            running = 0.0
            step_no_improve = 0           # reset each epoch
            best_in_epoch = float('inf')
            pb = tqdm(train_loader, leave=False, desc=f"Train {epoch}")
            for batch in pb:
                batch = batch.to(device)

                with record_function("Forward"):
                    logits = model(batch[:, :-1])

                with record_function("ComputeLoss"):
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        batch[:, 1:].reshape(-1),
                    )

                with record_function("Backward"):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                running += loss.item()
                pb.set_postfix(loss=loss.item())
                prof.step()
                global_step += 1

                # ── quick validation ─────────────────────────────────
                if global_step % val_every_steps == 0:
                    q_val = evaluate(model, val_loader, device, loss_fn, val_subset)
                    pb.write(f"[q-val] step {global_step:,}: loss = {q_val:.4f}")
                    if q_val < best_in_epoch - 0.01:
                        best_in_epoch = q_val
                        step_no_improve = 0
                    else:
                        step_no_improve += 1
                        if step_no_improve >= step_patience:
                            pb.write(f"Early stop (in epoch): no improvement after {step_patience} checks")
                            break

            # ── full validation at epoch end ────────────────────────────────
            full_val = evaluate(model, val_loader, device, loss_fn, 1.0)
            val_curve.append(full_val)
            train_curve.append(running / len(pb))

            print(f"Epoch {epoch}/{num_epochs} | train={train_curve[-1]:.4f} | val={full_val:.4f}")

            epoch_duration = time.time() - epoch_start_time  #Added profiling support related
            print(f"Epoch {epoch} ── wall time {epoch_duration:.1f}s")  #Added profiling support related

            if full_val < best_across_epochs - 0.1:
                best_in_epoch = full_val
                epoch_no_improve = 0
            else:
                epoch_no_improve += 1
                if epoch_no_improve >= patience:
                    print(f"Early stop: no improvement for {patience} epochs")
                    break

            # ── checkpoint every 5 epochs ───────────────────────────
            if epoch % 5 == 0:
                ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_epoch_{epoch}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    #Added profiling support related: summarise and write top N ops
    stats = prof.key_averages()
    summary = stats.table(sort_by="self_cpu_time_total", row_limit=profile_top_n)

    # write standalone summary.txt under Profilerlogs/run_<timestamp>/
    txt_path = os.path.join(CKPT_DIR, "Profilerlogs", f"run_{timestamp}", "summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary)

    # embed summary in TensorBoard under the "Text" tab
    writer.add_text("profiling/summary", summary)
    writer.flush()
    writer.close()

    return train_curve, val_curve

