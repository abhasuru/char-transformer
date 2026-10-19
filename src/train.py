"""
Stage 4-5: train one transformer, save the model, curves, attention maps and a sample.

Run:  python -m src.train                       # defaults: context 64, 4 layers
      python -m src.train --block_size 32 --steps 5000
"""

import argparse
import json
import os
from dataclasses import asdict

import torch

from src.data import get_device, load_dataset
from src.model import GPT, GPTConfig, count_params, generate
from src.plots import plot_attention, plot_loss_curve
from src.training import TrainConfig, train


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="results/transformer")
    return p


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    tok, train_data, val_data = load_dataset()
    mcfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
                     n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout)
    tcfg = TrainConfig(max_steps=args.steps, batch_size=args.batch_size, block_size=args.block_size,
                       lr=args.lr, device=device)
    model = GPT(mcfg)
    print(f"device: {device} | parameters: {count_params(model):,}")

    history = train(model, train_data, val_data, tcfg)

    torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg), "chars": tok.chars},
               os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_loss_curve(history, os.path.join(args.out, "loss_curve.png"),
                    title=f"Transformer, context {args.block_size}")

    # Attention maps on a short familiar string.
    probe = "To be, or not to be"[: args.block_size]
    model.eval()
    with torch.no_grad():
        model(torch.tensor([tok.encode(probe)], device=device))
    plot_attention(model.attention_maps(), probe, os.path.join(args.out, "attention_layer0.png"), layer=0)
    plot_attention(model.attention_maps(), probe,
                   os.path.join(args.out, f"attention_layer{args.n_layer - 1}.png"), layer=args.n_layer - 1)

    start = torch.tensor([tok.encode("ROMEO:")], device=device)
    sample = tok.decode(generate(model, start, 500, temperature=0.8)[0].tolist())
    with open(os.path.join(args.out, "sample.txt"), "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"\nfinal val loss: {history['val_loss'][-1]:.4f}")
    print(f"saved to {args.out}/")
    print("\n--- sample ---")
    print(sample)


if __name__ == "__main__":
    main()
