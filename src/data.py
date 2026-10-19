"""
Data loading, tokenization, and batching.

We train on Tiny Shakespeare (~1.1M characters). Each unique character becomes
one token, so the "vocabulary" is just the ~65 distinct characters in the text.
"""

import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")


def load_text(path=DATA_PATH):
    """Read the training text, downloading it the first time."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """Maps each character to an integer id and back.

    Real LLMs use subword tokenizers (e.g. BPE) so a token is often a whole word
    or word piece. Characters keep the vocabulary tiny and the idea obvious.
    """

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def load_dataset(val_frac=0.1):
    """Return (tokenizer, train_data, val_data) as 1-D tensors of token ids.

    The split is contiguous (first 90% train, last 10% val) rather than random,
    so validation text is genuinely unseen passages, not shuffled neighbours of
    training text.
    """
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * (1 - val_frac))
    return tok, data[:n], data[n:]


def get_batch(data, block_size, batch_size, device="cpu", generator=None):
    """Sample a random batch of (input, target) sequences.

    x is a window of `block_size` characters; y is the same window shifted one
    to the right. So for every position t, y[t] is "the character after x[:t+1]".
    One sequence of length T therefore gives T training examples at once.

        text:  T o _ b e
        x:     T o _ b
        y:     o _ b e
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
