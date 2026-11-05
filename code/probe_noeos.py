"""probe_noeos.py -- is the PE gap caused by the stop decision?

Decomposition (Newman et al. 2020): exact match conflates *what the model
computes* with *where it stops*. Our fixed-length runs show the two are separable:
with an explicit code the model emits EOS at exactly L_tr with probability 1.0,
while NoPE and ALiBi stop at the true length. Under an oracle length, RoPE
recovers 0.99 token accuracy -- its failure is almost entirely the stop decision.

That suggests a causal test. Train without any EOS token at all: every target is a
content token, so there is no boundary to place. If the PE gap on exact match is
mostly the stop decision, the gap should largely disappear in this condition.

    python probe_noeos.py --pes nope,sin,rope --seeds 0,1
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from arlib import VALUES, TinyGPT
from padded import K, PAD, T, VOCAB_P

LENS = (8, 10, 12, 16)


def labels(task, x, L):
    if task == "copy":
        return x.clone()
    if task == "reverse":
        return x.flip(1)
    if task == "sort":
        return torch.sort(x, dim=1).values
    raise ValueError(task)


def make_noeos(task, L, n, gen):
    x = torch.randint(0, VALUES, (n, L), generator=gen)
    y = labels(task, x, L)
    pad1 = torch.full((n, K - L), PAD, dtype=torch.long)
    sep = torch.full((n, 1), 10, dtype=torch.long)          # SEP == VALUES
    pad2 = torch.full((n, T - K - L - 1), PAD, dtype=torch.long)
    seq = torch.cat([x, pad1, sep, y, pad2], dim=1)          # no EOS token at all
    inp, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    mask[:, K:K + L] = True                                  # content only
    return inp, tgt, mask, y


@torch.no_grad()
def evaluate(model, task, L, n=512, seed=777):
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed + L)
    inp, _, _, y = make_noeos(task, L, n, g)
    pred = model(inp.to(dev))[:, K:K + L].argmax(-1).cpu()
    hit = (pred == y).float()
    return {"seq": round(hit.all(1).float().mean().item(), 4),
            "tok": round(hit.mean().item(), 4),
            "per_pos": [round(v, 3) for v in hit.mean(0).tolist()]}


def train(task, pe, seed, steps=2000, L_train=8, d=256, nlayer=4, nhead=8,
          batch=128, lr=5e-4, wd=1e-2, ckpt=None):
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

    for step in range(1, steps + 1):
        inp, tgt, mask, _ = make_noeos(task, L_train, batch, g)
        inp, tgt, mask = inp.to(dev), tgt.to(dev), mask.to(dev)
        loss = F.cross_entropy(model(inp)[mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save({"state": model.state_dict(),
                    "cfg": {"d": d, "nlayer": nlayer, "nhead": nhead, "pe": pe}}, ckpt)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="copy")
    ap.add_argument("--pes", default="nope,sin,rope")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--outdir", default="../results/noeos")
    ap.add_argument("--ckptdir", default="../results/ckpt")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    todo = [(p, s) for p in a.pes.split(",") for s in a.seeds.split(",")
            if not os.path.exists(os.path.join(a.outdir, f"{a.task}_{p}_s{s}.json"))]
    print(f"{len(todo)} runs queued", flush=True)

    for i, (pe, s) in enumerate(todo, 1):
        t0 = time.time()
        m = train(a.task, pe, int(s), steps=a.steps,
                  ckpt=os.path.join(a.ckptdir, f"ne_{a.task}_{pe}_s{s}.pt"))
        res = {str(L): evaluate(m, a.task, L) for L in LENS}
        json.dump({"task": a.task, "pe": pe, "seed": int(s), "eos": False,
                   "L": res}, open(os.path.join(a.outdir, f"{a.task}_{pe}_s{s}.json"), "w"),
                  indent=1)
        print(f"[{i}/{len(todo)}] {pe:6s} s{s}  "
              + " ".join(f"L{L}={res[str(L)]['seq']:.2f}/{res[str(L)]['tok']:.2f}" for L in LENS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
