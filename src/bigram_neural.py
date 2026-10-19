"""
Stage 2: the bigram model again, but learned by gradient descent.

It sees exactly as much as the counting model (one character), so the best it
can do is match the counting model's loss. The point is the machinery: if this
learns to ~the same number, the loss/backprop/optimizer loop is working, and we
can trust it on the transformer.

Run:  python -m src.bigram_neural
"""

import argparse

import torch

from src.data import get_device, load_dataset
from src.model import BigramLanguageModel, generate
from src.training import TrainConfig, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()

    torch.manual_seed(1337)
    device = get_device()
    tok, train_data, val_data = load_dataset()

    model = BigramLanguageModel(tok.vocab_size)
    # A lookup table learns fine with a large learning rate; no warmup or weight decay
    # needed. block_size=8 just means each batch gives 8 examples per sequence.
    cfg = TrainConfig(max_steps=args.steps, batch_size=32, block_size=8, lr=1e-1,
                      warmup_steps=0, weight_decay=0.0, eval_interval=500, device=device)
    print(f"device: {device}")
    history = train(model, train_data, val_data, cfg)

    print(f"\nfinal val loss: {history['val_loss'][-1]:.4f}  (compare with the counting bigram)")
    start = torch.full((1, 1), tok.stoi["\n"], dtype=torch.long, device=device)
    print("\n--- sample ---")
    print(tok.decode(generate(model, start, 300)[0].tolist()))


if __name__ == "__main__":
    main()
