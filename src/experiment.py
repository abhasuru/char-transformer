"""
Stage 6: the experiment. How much does seeing more previous text help?

Train the same transformer with context lengths 8, 16, 32, 64 and compare
validation loss. Everything except block_size is held fixed, including:

  * the number of characters trained on. A batch of B sequences of length T
    contains B*T prediction targets, so with a fixed batch size the context-64
    model would get 8x more training signal per step than context-8, and we
    couldn't tell "more context" apart from "more data". Instead we fix
    tokens_per_step and set batch_size = tokens_per_step / block_size.
  * steps, learning-rate schedule, model width/depth, and the eval text seed.

Two numbers per model:
  val_loss       - average over all positions in the window. Early positions
                   have little context, so this understates what long-context
                   models can do.
  last_pos_loss  - loss at the final position only, where the model has its full
                   context. This is the cleanest "what does context N buy you".

Run:  python -m src.experiment                      # one seed per context length
      python -m src.experiment --seeds 1 2 3        # mean +/- std across seeds
      python -m src.experiment --quick              # tiny CPU smoke test
"""

import argparse
import csv
import json
import os
import statistics

import torch

from src.data import get_device, load_dataset
from src.model import GPT, GPTConfig, count_params
from src.plots import plot_context_results, plot_per_position
from src.training import TrainConfig, estimate_loss, train


def run_one(block_size, seed, args, tok, train_data, val_data, device):
    torch.manual_seed(seed)
    batch_size = args.tokens_per_step // block_size
    mcfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size, n_embd=args.n_embd,
                     n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout)
    tcfg = TrainConfig(max_steps=args.steps, batch_size=batch_size, block_size=block_size,
                       lr=args.lr, eval_interval=args.eval_interval, eval_iters=args.eval_iters,
                       device=device)
    model = GPT(mcfg)
    print(f"\n=== context {block_size} | seed {seed} | batch {batch_size} | params {count_params(model):,} ===")
    history = train(model, train_data, val_data, tcfg)

    # Final, more precise evaluation on a fixed set of validation windows.
    val_loss, per_pos = estimate_loss(model, val_data, block_size, batch_size,
                                      args.final_eval_iters, device, seed=123)
    return {
        "block_size": block_size,
        "seed": seed,
        "batch_size": batch_size,
        "params": count_params(model),
        "val_loss": val_loss,
        "last_pos_loss": per_pos[-1].item(),
        "train_loss": history["train_loss"][-1],
        "train_time_s": history["train_time_s"],
        "per_position": per_pos.tolist(),
    }


def summarize(runs):
    rows = []
    for bs in sorted({r["block_size"] for r in runs}):
        rs = [r for r in runs if r["block_size"] == bs]
        row = {"block_size": bs, "n_seeds": len(rs), "params": rs[0]["params"]}
        for key in ["val_loss", "last_pos_loss", "train_loss", "train_time_s"]:
            vals = [r[key] for r in rs]
            row[f"{key}_mean"] = statistics.mean(vals)
            row[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def write_table(rows, path):
    lines = ["| context | val loss (all positions) | val loss (last position) | train loss | gap (val - train) | train time |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        pm = lambda k: f"{r[k + '_mean']:.4f}" + (f" ± {r[k + '_std']:.4f}" if r["n_seeds"] > 1 else "")
        gap = r["val_loss_mean"] - r["train_loss_mean"]
        lines.append(f"| {r['block_size']} | {pm('val_loss')} | {pm('last_pos_loss')} | {pm('train_loss')} "
                     f"| {gap:+.4f} | {r['train_time_s_mean']:.0f}s |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block_sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--seeds", type=int, nargs="+", default=[1337])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--tokens_per_step", type=int, default=4096)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--final_eval_iters", type=int, default=200)
    p.add_argument("--out", type=str, default="results/context_experiment")
    p.add_argument("--quick", action="store_true", help="tiny run to check the pipeline works")
    args = p.parse_args()
    if args.quick:
        args.steps, args.eval_interval, args.eval_iters, args.final_eval_iters = 60, 30, 2, 5
        args.n_embd, args.n_layer = 32, 2
        args.tokens_per_step = 1024

    device = get_device()
    os.makedirs(args.out, exist_ok=True)
    tok, train_data, val_data = load_dataset()
    print(f"device: {device}")

    runs = [run_one(bs, seed, args, tok, train_data, val_data, device)
            for bs in args.block_sizes for seed in args.seeds]
    rows = summarize(runs)

    with open(os.path.join(args.out, "runs.json"), "w") as f:
        json.dump({"args": vars(args), "runs": runs}, f, indent=2)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    table = write_table(rows, os.path.join(args.out, "results.md"))
    plot_context_results(rows, os.path.join(args.out, "context_vs_loss.png"))

    # Per-position curves, averaged over seeds.
    per_pos = {}
    for bs in args.block_sizes:
        curves = [r["per_position"] for r in runs if r["block_size"] == bs]
        per_pos[bs] = [statistics.mean(vals) for vals in zip(*curves)]
    plot_per_position(per_pos, os.path.join(args.out, "loss_by_position.png"))

    print("\n" + table)
    print(f"\nsaved results and plots to {args.out}/")


if __name__ == "__main__":
    main()
