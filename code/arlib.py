"""arlib.py -- tiny decoder-only Transformer + length-generalization task suite.

Deliberately small enough to train on CPU. Everything here is a single
self-contained module so that a run is reproducible from (task, pe, seed).

Sequence layout (all tasks are sequence-to-sequence):
    x_1 .. x_L  SEP  y_1 .. y_L  EOS
Loss and exact-match accuracy are computed on y_1..y_L only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VALUES = 10
SEP = VALUES
EOS = VALUES + 1
VOCAB = VALUES + 2

# Task order follows the *length-coupling* axis:
#   position map independent of L : copy, prefix_xor, sort
#   position map depends on L     : swap_halves, reverse
TASKS = ("copy", "prefix_xor", "sort", "swap_halves", "reverse")
PES = ("nope", "learned", "sin", "rope", "alibi")

COUPLED = {"swap_halves", "reverse"}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def make_data(task, L, n, gen):
    """Return (inp, tgt, loss_mask, y) for a batch of n sequences of length L."""
    x = torch.randint(0, VALUES, (n, L), generator=gen)
    if task == "copy":                      # y_i = x_i              (local)
        y = x.clone()
    elif task == "prefix_xor":              # y_i = x_1 ^ ... ^ x_i  (local, causal)
        x = x % 2
        y = torch.cumsum(x, dim=1) % 2
    elif task == "reverse":                 # y_i = x_{L+1-i}        (L-coupled)
        y = x.flip(1)
    elif task == "sort":                    # y = sorted(x)          (position-free)
        y = torch.sort(x, dim=1).values
    elif task == "swap_halves":             # y = x[L/2:] ++ x[:L/2] (L-coupled)
        if L % 2:
            raise ValueError("swap_halves requires even L")
        h = L // 2
        y = torch.cat([x[:, h:], x[:, :h]], dim=1)
    else:
        raise ValueError(f"unknown task {task}")

    sep = torch.full((n, 1), SEP, dtype=torch.long)
    eos = torch.full((n, 1), EOS, dtype=torch.long)
    seq = torch.cat([x, sep, y, eos], dim=1)

    inp = seq[:, :-1]
    tgt = seq[:, 1:]
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    mask[:, -(L + 1):] = True                     # y_1..y_L and EOS
    return inp, tgt, mask, y


# --------------------------------------------------------------------------
# positional encodings
# --------------------------------------------------------------------------
def _sinusoid(max_len, d):
    pos = torch.arange(max_len).float().unsqueeze(1)
    i = torch.arange(0, d, 2).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _rope(q, k, pos):
    """Rotary position embedding applied to q, k of shape [B, H, T, dh]."""
    dh = q.shape[-1]
    inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=q.device).float() / dh))
    freqs = pos.float().unsqueeze(-1) * inv              # [T, dh/2]
    cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

    return rot(q), rot(k)


def _alibi(T, H, device):
    slopes = 2.0 ** (-8.0 * (torch.arange(1, H + 1, device=device).float()) / H)
    dist = (torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :]).abs().float()
    return -slopes[:, None, None] * dist[None]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class Attn(nn.Module):
    def __init__(self, d, h, pe):
        super().__init__()
        self.h, self.dh, self.pe = h, d // h, pe
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, pos):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if self.pe == "rope":
            q, k = _rope(q, k, pos)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        if self.pe == "alibi":
            att = att + _alibi(T, self.h, x.device)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, d, h, pe):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, h, pe)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pos):
        x = x + self.attn(self.ln1(x), pos)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d=128, nlayer=3, nhead=4, pe="nope", max_len=256,
                 vocab=VOCAB):
        super().__init__()
        assert pe in PES, pe
        self.pe = pe
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d) if pe == "learned" else None
        if pe == "sin":
            self.register_buffer("sin_pe", _sinusoid(max_len, d), persistent=False)
        self.blocks = nn.ModuleList([Block(d, nhead, pe) for _ in range(nlayer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx)
        if self.pe == "learned":
            x = x + self.pos(pos)
        elif self.pe == "sin":
            x = x + self.sin_pe[pos]
        for b in self.blocks:
            x = b(x, pos)
        return self.head(self.ln_f(x))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, task, L, n=512, seed=12345):
    """Exact-match accuracy on length L (independent generator => no train-set leak)."""
    was_training = model.training
    dev = next(model.parameters()).device
    model.eval()
    g = torch.Generator().manual_seed(seed + L)
    inp, _, mask, y = make_data(task, L, n, g)
    inp, y = inp.to(dev), y.to(dev)
    # supervised positions are y_1..y_L then EOS; take the first L of them
    idx = mask[0].nonzero().flatten()[:L]
    pred = model(inp)[:, idx].argmax(-1)
    acc = (pred == y).all(dim=1).float().mean().item()
    if was_training:
        model.train()
    return acc
