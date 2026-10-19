"""
Sanity checks that catch the classic transformer bugs. Run:  python -m pytest -q
"""

import math

import torch

from src.model import GPT, BigramLanguageModel, GPTConfig

V = 65


def small_model(block_size=16):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=V, block_size=block_size, n_embd=32, n_head=4, n_layer=2, dropout=0.0)).eval()


def test_output_shapes():
    model = small_model()
    idx = torch.randint(V, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, V)
    assert loss.dim() == 0


def test_initial_loss_is_near_uniform():
    # A fresh model should be unsure of everything: loss ~= ln(V) = 4.17.
    model = small_model()
    idx = torch.randint(V, (8, 16))
    _, loss = model(idx, torch.randint(V, (8, 16)))
    assert abs(loss.item() - math.log(V)) < 0.2


def test_causal_mask_blocks_the_future():
    # Changing the token at position 10 must not change predictions at positions 0..9.
    # If it does, the model is peeking at the answer during training.
    model = small_model()
    a = torch.randint(V, (1, 16))
    b = a.clone()
    b[0, 10] = (a[0, 10] + 1) % V
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-6)
    assert not torch.allclose(la[0, 10:], lb[0, 10:])


def test_attention_rows_sum_to_one_and_ignore_future():
    model = small_model()
    model(torch.randint(V, (2, 16)))
    w = model.attention_maps()[0][0]  # layer 0, head 0: (B, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(2, 16), atol=1e-5)
    assert torch.all(torch.triu(w[0], diagonal=1) == 0)


def test_can_overfit_one_batch():
    # If a model can't memorise a single batch, something is broken (wiring, lr, masking).
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=V, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0))
    x = torch.randint(V, (4, 16))
    y = torch.randint(V, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_bigram_interface():
    model = BigramLanguageModel(V)
    logits, loss = model(torch.randint(V, (2, 8)), torch.randint(V, (2, 8)))
    assert logits.shape == (2, 8, V) and loss is not None
