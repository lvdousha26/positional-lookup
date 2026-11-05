"""padded.py -- decouple the position range from the effective length.

In the standard format a sequence of L tokens occupies 2L+1 positions, so
"the model runs out of trained positions" and "the model runs out of trained
lengths" are the same event: both happen exactly at L > L_tr. The cliff probe in
the paper therefore cannot distinguish them.

This module fixes the total token count at T=41 (all positions 0..40 present in
every example, the tail filled with PAD) and pins <SEP> at K=21. Only the number
of *effective* tokens varies. Every position is seen during training; if accuracy
still collapses at L_tr+2, the solution is indexed by length, not by position.

    x(0..L-1) | PAD(L..K-1) | SEP(K) | y(K+1..K+L) | EOS | PAD(..T-1)
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from arlib import COUPLED, EOS, SEP, VALUES, TinyGPT

PAD = 12
VOCAB_P = 13
T = 41                      # fixed total tokens
K = 21                      # fixed <SEP> position


def labels(task, x, L):
    if task == "copy":
        return x.clone()
    if task == "prefix_xor":
        xx = x % 2
        return torch.cumsum(xx, dim=1) % 2
    if task == "reverse":
        return x.flip(1)
    if task == "sort":
        return torch.sort(x, dim=1).values
    if task == "swap_halves":
        h = L // 2
        return torch.cat([x[:, h:], x[:, :h]], dim=1)
    raise ValueError(task)


def make_padded(task, L, n, gen):
    x = torch.randint(0, VALUES, (n, L), generator=gen)
    y = labels(task, x, L)
    pad1 = torch.full((n, K - L), PAD, dtype=torch.long)
    sep = torch.full((n, 1), SEP, dtype=torch.long)
    eos = torch.full((n, 1), EOS, dtype=torch.long)
    pad2 = torch.full((n, T - K - L - 2), PAD, dtype=torch.long)
    seq = torch.cat([x, pad1, sep, y, eos, pad2], dim=1)
    inp, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    mask[:, K:K + L + 1] = True          # y_1..y_L then EOS
    return inp, tgt, mask, y


@torch.no_grad()
def evaluate_padded(model, task, L, n=512, seed=1234):
    was = model.training
    dev = next(model.parameters()).device
    model.eval()
    g = torch.Generator().manual_seed(seed + L)
    inp, _, _, y = make_padded(task, L, n, g)
    pred = model(inp.to(dev))[:, K:K + L].argmax(-1)
    acc = (pred.cpu() == y).all(dim=1).float().mean().item()
    if was:
        model.train()
    return acc


def train(task, pe, seed, steps=2000, L_train=8, eval_lens=(8, 10, 12, 16),
          d=256, nlayer=4, nhead=8, batch=128, lr=5e-4, wd=1e-2):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    model = TinyGPT(d=d, nlayer=nlayer, nhead=nhead, pe=pe, vocab=VOCAB_P).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    curve, t0 = [], time.time()
    for step in range(1, steps + 1):
        inp, tgt, mask, _ = make_padded(task, L_train, batch, g)
        inp, tgt, mask = inp.to(dev), tgt.to(dev), mask.to(dev)
        loss = F.cross_entropy(model(inp)[mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 250 == 0 or step == steps:
            rec = {"step": step, "loss": round(loss.item(), 4)}
            for L in eval_lens:
                rec["L%d" % L] = round(evaluate_padded(model, task, L), 4)
            curve.append(rec)
            cells = " ".join(f"{k}={v}" for k, v in rec.items() if k != "step")
            print(f"[{task:10s} {pe:8s} s{seed} pad] {step:5d}  {cells}", flush=True)

    return {"task": task, "pe": pe, "seed": seed, "pad": True, "T": T, "K": K,
            "steps": steps, "sec": round(time.time() - t0, 1), "curve": curve}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="copy")
    ap.add_argument("--pe", default="sin")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = train(a.task, a.pe, a.seed, a.steps)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    print("final:", r["curve"][-1], f"({r['sec']}s)")
