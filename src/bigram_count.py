"""
Stage 1: a counting bigram model. No neural network, no training loop.

For every pair of adjacent characters (a, b) in the training text, count how
often b follows a. Normalise each row of counts into probabilities and you have
P(next char | current char). This is the baseline every later model must beat.

Run:  python -m src.bigram_count
"""

import math

import torch

from src.data import load_dataset


def main():
    torch.manual_seed(1337)
    tok, train_data, val_data = load_dataset()
    V = tok.vocab_size

    # Count all bigrams at once. Pair (a, b) maps to the single index a*V + b,
    # bincount tallies them, and we reshape back into a V x V table.
    pair_ids = train_data[:-1] * V + train_data[1:]
    counts = torch.bincount(pair_ids, minlength=V * V).view(V, V).float()

    # Add-one (Laplace) smoothing: a pair never seen in training would otherwise
    # get probability 0, and -log(0) is infinite loss on validation.
    probs = (counts + 1) / (counts + 1).sum(dim=1, keepdim=True)

    # Validation loss = average negative log-likelihood of the true next char.
    # This is exactly the cross-entropy loss the neural models will report.
    val_nll = -torch.log(probs[val_data[:-1], val_data[1:]]).mean().item()
    train_nll = -torch.log(probs[train_data[:-1], train_data[1:]]).mean().item()

    print(f"vocab size:                  {V}")
    print(f"uniform-guess loss  ln({V}):  {math.log(V):.4f}   <- knowing nothing")
    print(f"count bigram train loss:     {train_nll:.4f}")
    print(f"count bigram val loss:       {val_nll:.4f}   <- the baseline to beat")

    # Show what the model 'knows': most likely characters after a few inputs.
    for ch in ["q", "t", " "]:
        top = torch.topk(probs[tok.stoi[ch]], 3)
        pretty = ", ".join(f"{tok.itos[int(i)]!r}:{p:.2f}" for p, i in zip(top.values, top.indices))
        print(f"after {ch!r} -> {pretty}")

    # Generate by repeatedly sampling the next char from the current char's row.
    idx = tok.stoi["\n"]
    out = []
    for _ in range(300):
        idx = torch.multinomial(probs[idx], num_samples=1).item()
        out.append(idx)
    print("\n--- sample ---")
    print(tok.decode(out))


if __name__ == "__main__":
    main()
