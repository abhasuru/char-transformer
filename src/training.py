"""
The training loop and evaluation, shared by every neural model.

One training step:
  1. forward:   run a batch through the model, get the loss (how wrong it was)
  2. backward:  loss.backward() computes d(loss)/d(parameter) for every
                parameter via backpropagation (the chain rule, applied layer by
                layer from the loss back to the inputs)
  3. update:    the optimizer nudges each parameter against its gradient
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.data import get_batch


@dataclass
class TrainConfig:
    max_steps: int = 3000
    batch_size: int = 64
    block_size: int = 64
    lr: float = 1e-3
    min_lr_frac: float = 0.1     # cosine schedule decays to lr * this
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    device: str = "cpu"


def lr_at(step, cfg):
    """Linear warmup, then cosine decay.

    Warmup: at step 0 the weights are random and gradients are large and noisy,
    so big steps can knock the model somewhere bad. Ramp up gently.
    Cosine decay: large steps early to make fast progress, small steps late to
    settle into a minimum instead of bouncing around it.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    min_lr = cfg.lr * cfg.min_lr_frac
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device, seed=0):
    """Average loss over `eval_iters` batches, plus the loss at each position.

    The batches come from a generator with a fixed seed, so every model (and every
    eval during training) is scored on exactly the same text. Without that, part of
    the difference between two runs would be which batches they happened to get.

    Per-position loss: position t has only t characters of context before it, so
    per_pos[t] answers "how good is the model with t+1 characters to look at?"

    model.eval() switches off dropout so evaluation is deterministic.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    per_pos = torch.zeros(block_size)
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device, generator=g)
        logits, _ = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="none")
        per_pos += loss.view(B, T).mean(dim=0).cpu()
    model.train()
    per_pos /= eval_iters
    return per_pos.mean().item(), per_pos


def train(model, train_data, val_data, cfg, log=print):
    """Train `model` and return a history dict of losses over time."""
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}
    start = time.time()

    for step in range(cfg.max_steps + 1):
        # Evaluate on a schedule (and at the very end).
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            tr, _ = estimate_loss(model, train_data, cfg.block_size, cfg.batch_size, cfg.eval_iters, cfg.device)
            va, _ = estimate_loss(model, val_data, cfg.block_size, cfg.batch_size, cfg.eval_iters, cfg.device)
            history["step"].append(step)
            history["train_loss"].append(tr)
            history["val_loss"].append(va)
            history["lr"].append(lr_at(step, cfg))
            log(f"step {step:5d} | train {tr:.4f} | val {va:.4f} | lr {lr_at(step, cfg):.2e} | {time.time() - start:6.1f}s")
        if step == cfg.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
        _, loss = model(x, y)

        # A NaN/inf loss means training has diverged (usually lr too high or a
        # numerical bug). Stop loudly instead of silently training garbage.
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became {loss.item()} at step {step}; try a lower learning rate")

        optimizer.zero_grad(set_to_none=True)  # gradients accumulate by default; clear last step's
        loss.backward()
        # Gradient clipping: if one unlucky batch produces a huge gradient, scale it
        # down so a single step cannot blow up the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    history["train_time_s"] = time.time() - start
    return history
