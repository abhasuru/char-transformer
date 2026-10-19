"""
The models, from simplest to the full transformer.

Every model has the same interface:
    logits, loss = model(idx, targets)
where idx is (B, T) token ids, logits is (B, T, vocab_size) - a score for every
possible next character at every position - and loss is the cross-entropy
against targets (or None if no targets were given).

Shape letters used throughout:
    B = batch size, T = sequence length (time), C = embedding size (channels),
    V = vocab size, H = head size
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def lm_loss(logits, targets):
    """Cross-entropy over every position. F.cross_entropy wants (N, V) and (N,)."""
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


# ---------------------------------------------------------------------------
# Stage 2: neural bigram
# ---------------------------------------------------------------------------

class BigramLanguageModel(nn.Module):
    """A V x V table of learnable scores: row = current char, row values = logits
    for the next char. It is the counting model's table, except gradient descent
    fills it in instead of counting. It still only sees one character.
    """

    block_size = 1  # how many characters it can use as context

    def __init__(self, vocab_size):
        super().__init__()
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_logits(idx)  # (B, T, V): a row lookup per character
        loss = None if targets is None else lm_loss(logits, targets)
        return logits, loss


# ---------------------------------------------------------------------------
# Stages 3-4: attention and the transformer
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64   # context length: max characters the model can look back over
    n_embd: int = 128      # size of each character's vector
    n_head: int = 4        # attention heads per block
    n_layer: int = 4       # number of transformer blocks stacked
    dropout: float = 0.1


class Head(nn.Module):
    """One head of causal self-attention.

    Each position emits three vectors from its current representation:
      query - "what am I looking for?"
      key   - "what do I contain?"
      value - "what will I pass along if you pick me?"
    Position t scores every earlier position s by query_t . key_s, softmaxes the
    scores into weights that sum to 1, and outputs the weighted average of the
    values. The causal mask sets scores for s > t to -inf so the future gets
    weight 0 - otherwise the model could read the answer during training.
    """

    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # Lower-triangular matrix of ones: row t has ones in columns 0..t.
        # A buffer moves with .to(device) but is not a trained parameter.
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)
        self.last_attn = None  # kept for visualising what the head looks at

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)    # (B, T, H)
        v = self.value(x)  # (B, T, H)

        # Scores: (B, T, H) @ (B, H, T) -> (B, T, T). Entry [t, s] = how much t wants s.
        # Divide by sqrt(H): dot products of H-dim vectors grow like sqrt(H); without
        # scaling, softmax saturates into a near one-hot and gradients vanish.
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # each row sums to 1 over positions <= t
        self.last_attn = weights.detach()
        weights = self.dropout(weights)
        return weights @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, each free to track a different relationship
    (e.g. one head watches the previous character, another the start of the word).
    Their outputs are concatenated and mixed back to n_embd by a linear layer.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly across heads"
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_head * H) = (B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A small MLP applied to each position independently.

    Attention is communication (positions gather information from each other);
    the feed-forward layer is computation (each position processes what it gathered).
    The 4x hidden width is the convention from the original transformer paper.
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention, then feed-forward, each wrapped in a
    residual connection with layer norm applied first ("pre-norm", as in GPT-2).

    Residual: x = x + f(x). Each sub-layer adds an adjustment instead of replacing
    x, so there is always a direct path for gradients from the loss back to the
    early layers - that is what lets deep stacks train.
    LayerNorm: rescales each position's vector to mean 0, variance 1 (then a
    learned scale/shift), keeping activations in a stable range as depth grows.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """A decoder-only transformer language model.

    tokens -> token embedding + position embedding -> N blocks -> final LayerNorm
           -> linear layer to V logits
    Attention by itself has no notion of order (it is a weighted average over a
    set), so the position embedding is what tells the model "this char is 3 back".
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Small random weights (std 0.02, as in GPT-2) so the initial logits are near
        # zero, the initial prediction is near uniform, and the starting loss is
        # about ln(V). A much higher starting loss means the init is too confident.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C) + (T, C) broadcasts
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        loss = None if targets is None else lm_loss(logits, targets)
        return logits, loss

    def attention_maps(self):
        """Attention weights from the last forward pass: list[layer][head] of (B, T, T)."""
        return [[h.last_attn for h in blk.attn.heads] for blk in self.blocks]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    """Autoregressive sampling: predict, sample one char, append, repeat.

    Only the last block_size characters are fed in, because that is all the model
    was built to see. Temperature < 1 sharpens the distribution (safer, more
    repetitive), > 1 flattens it (more random).
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # only the prediction after the last char
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return idx
