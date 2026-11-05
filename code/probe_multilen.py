"""probe_multilen.py -- does the solution key on the *set of trained lengths*?

The padded protocol rules out "position out of range": every one of the 41
positions occurs in every training example, yet learned and sinusoidal codes
still collapse at the first length beyond training. What survives is the
length-keyed-lookup account: the model fits a separate solution per observed
length, and has nothing to fall back on at a length it never saw.

That account makes a prediction the position account does not. Train on
{6, 8, 10} instead of {8}. A model that has abstracted a positional *rule* should
either extrapolate to 12 or at least fail differently; a model that has fitted
one entry per length should behave perfectly at 6, 8, 10 and collapse at 12,
i.e. the failure should track the last *seen* length rather than the largest
*position*.

Usage:
    python probe_multilen.py --tasks copy --pes sin,nope --seeds 0,1
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from arlib import TinyGPT
from padded import K, VOCAB_P, evaluate_padded, make_padded

EVAL = (4, 6, 8, 10, 12, 16)


def train_multilen(task, pe, seed, lens=(6, 8, 10), steps=2000,
                   d=256, nlayer=4, nhead=8, batch=128, lr=5e-4, wd=1e-2,
                   ckpt=None, verbose=False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    model = TinyGPT(d=d, nlayer=nlayer, nhead=nhead, pe=pe, vocab=VOCAB_P).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    curve, t0 = [], time.time()
    for step in range(1, steps + 1):
        # sample the length uniformly, so no single length can dominate the fit
        L = lens[int(torch.randint(len(lens), (1,), generator=g))]
        inp, tgt, mask, _ = make_padded(task, L, batch, g)
        inp, tgt, mask = inp.to(dev), tgt.to(dev), mask.to(dev)
        loss = F.cross_entropy(model(inp)[mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 250 == 0 or step == steps:
            rec = {"step": step, "loss": round(loss.item(), 4)}
            for L in EVAL:
                rec["L%d" % L] = round(evaluate_padded(model, task, L), 4)
            curve.append(rec)
            if verbose:
                print(f"  {step:5d} " + " ".join(f"L{k[1:]}={v}" for k, v in rec.items()
                                                 if k.startswith("L")), flush=True)

    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save({"state": model.state_dict(),
                    "cfg": {"d": d, "nlayer": nlayer, "nhead": nhead, "pe": pe}}, ckpt)

    return {"task": task, "pe": pe, "seed": seed, "pad": True, "lens": list(lens),
            "steps": steps, "sec": round(time.time() - t0, 1), "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="copy")
    ap.add_argument("--pes", default="sin,nope")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lens", default="6,8,10")
    ap.add_argument("--single", default="8",
                    help="control regime: a single training length")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--outdir", default="../results/multilen")
    ap.add_argument("--ckptdir", default="../results/ckpt")
    a = ap.parse_args()

    regimes = {"single" + a.single: [int(a.single)],
               "multi" + a.lens.replace(",", "-"): [int(x) for x in a.lens.split(",")]}
    os.makedirs(a.outdir, exist_ok=True)

    todo = [(t, p, s, name, lens) for t in a.tasks.split(",")
            for p in a.pes.split(",") for s in [int(x) for x in a.seeds.split(",")]
            for name, lens in regimes.items()
            if not os.path.exists(os.path.join(a.outdir, f"{t}_{p}_{name}_s{s}.json"))]
    print(f"{len(todo)} runs queued", flush=True)

    for i, (task, pe, seed, name, lens) in enumerate(todo, 1):
        t0 = time.time()
        r = train_multilen(task, pe, seed, lens=lens, steps=a.steps,
                           ckpt=os.path.join(a.ckptdir, f"ml_{task}_{pe}_{name}_s{seed}.pt"))
        json.dump(r, open(os.path.join(a.outdir, f"{task}_{pe}_{name}_s{seed}.json"), "w"),
                  indent=1)
        c = r["curve"][-1]
        print(f"[{i:2d}/{len(todo)}] {task:6s} {pe:6s} {name:12s} s{seed}  "
              + " ".join(f"L{k[1:]}={v:.2f}" for k, v in c.items() if k.startswith("L"))
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
