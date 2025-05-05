import torch
from tqdm import tqdm
from torch import nn
from transformers import GPT2Tokenizer

# ──────────────────────────────────────────────────────────────────────────────── #
# CONSTANTS                                                                        #
#                                                                                  #
# Initialize GPT-2 tokenizer to retrieve pad_token_id                              #
# for masking tokens in the loss function.                                         #
# ──────────────────────────────────────────────────────────────────────────────── #

_tok = GPT2Tokenizer.from_pretrained("gpt2")
_tok.pad_token = _tok.eos_token
PAD_ID = _tok.pad_token_id

# ──────────────────────────────────────────────────────────────────────────────── #
# VALIDATION LOSS FUNCTION                                                         #
#                                                                                  #
# Computes mean cross-entropy loss over validation set or subset.                  #
# Used both for quick step-level validation and epoch-level validation.            #
# ──────────────────────────────────────────────────────────────────────────────── #

def evaluate(
    model,
    val_loader,
    device,
    loss_fn,
    subset_ratio: float = 1.0,
):
    """
    Evaluate model on validation data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: DataLoader for validation set.
        device: device string ("cpu" or "cuda").
        loss_fn: loss function (e.g., CrossEntropyLoss).
        subset_ratio: float in (0,1]; fraction of val_loader to use.

    Returns:
        Average validation loss over evaluated batches.
    """
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

# ──────────────────────────────────────────────────────────────────────────────── #
# TRAINING LOOP                                                                    #
#                                                                                  #
# Trains GPT-2 from scratch using custom PyTorch loop.                             #
# Implements two-tier early stopping:                                              #
#    • Step-level: quick-validation checks mid-epoch                               #
#    • Epoch-level: full validation at epoch end                                   #
#                                                                                  #
# Stops early if no improvement across configured patience thresholds.             #
# ──────────────────────────────────────────────────────────────────────────────── #

def train_summarizer(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    device="cpu",
    num_epochs: int = 20,
    patience: int = 4,              # epoch-level patience
    val_every_steps: int = 8000,  # quick-validation frequency
    step_patience: int = 4,         # patience for quick-validation
    val_subset: float = 0.2,        # fraction of val set for quick-check
):
    """
    Train GPT-2 summarizer with early stopping at both step and epoch levels.

    Args:
        model: PyTorch model (GPT-2 initialized from scratch).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: optimizer instance (e.g., Adam).
        scheduler: optional learning rate scheduler.
        device: target device ("cpu" or "cuda").
        num_epochs: maximum number of epochs.
        patience: epoch-level early stopping patience.
        val_every_steps: frequency (in steps) for quick validation.
        step_patience: step-level patience for consecutive non-improving quick validations.
        val_subset: fraction of validation set used during quick validation.

    Returns:
        Tuple of (train_loss_curve, val_loss_curve), each a list of floats per epoch.
    """

    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    train_curve, val_curve = [], []
    best_in_epoch = float("inf")
    epoch_no_improve = 0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        step_no_improve = 0           # reset each epoch
        
        best_in_epoch = float('inf')
        pb = tqdm(train_loader, leave=False, desc=f"Train {epoch}")
        for batch in pb:
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running += loss.item()
            pb.set_postfix(loss=loss.item())
            global_step += 1

            # ── quick validation ────────────────────────────────────────
            if global_step % val_every_steps == 0:
                q_val = evaluate(model, val_loader, device, loss_fn,
                                 val_subset)               # FIX
                pb.write(f"[q-val] step {global_step:,}: loss = {q_val:.4f}")

                if q_val < best_in_epoch - 0.01:
                    best_in_epoch = q_val
                    step_no_improve = 0
                else:
                    step_no_improve += 1
                    if step_no_improve >= step_patience:
                        print(f"[early-exit] epoch {epoch}: "
                              f"{step_patience} bad quick-vals")
                        break  # skip remainder of epoch

        # ── full validation at epoch end ────────────────────────────────
        full_val = evaluate(model, val_loader, device, loss_fn, 1.0)        # FIX
        val_curve.append(full_val)
        train_curve.append(running / len(pb))

        print(f"Epoch {epoch}/{num_epochs} | "
              f"train={train_curve[-1]:.4f} | val={full_val:.4f}")

        if full_val < best_across_epochs - 0.1:
            best_across_epochs = full_val
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= patience:
                print(f"Early stop: no improvement for {patience} epochs")
                break

    return train_curve, val_curve

