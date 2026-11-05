"""pilot.py -- single training run (device-agnostic) + CLI for one-off checks."""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from arlib import TinyGPT, evaluate, make_data

EVAL_LENS = (8, 10, 12, 16, 20)


def train_run(task, pe, seed, steps=4000, L_train=8, log_every=500,
              eval_lens=EVAL_LENS, d=256, nlayer=4, nhead=8,
              batch=128, lr=5e-4, wd=1e-2, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        # matmul-only TF32; accumulation stays fp32. Costs ~1e-3 relative
        # precision and buys ~2x throughput on this GPU.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    model = TinyGPT(d=d, nlayer=nlayer, nhead=nhead, pe=pe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )

    curve, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        inp, tgt, mask, _ = make_data(task, L_train, batch, g)
        inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
        loss = F.cross_entropy(model(inp)[mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps:
            rec = {"step": step, "loss": round(loss.item(), 4)}
            for L in eval_lens:
                rec["L%d" % L] = round(evaluate(model, task, L), 4)
            curve.append(rec)
            if verbose:
                cells = " ".join(f"{k}={v}" for k, v in rec.items() if k != "step")
                print(f"[{task:10s} {pe:8s} s{seed} {device}] {step:5d}  {cells}", flush=True)

    return {"task": task, "pe": pe, "seed": seed, "steps": steps, "device": device,
            "cfg": {"d": d, "nlayer": nlayer, "nhead": nhead, "batch": batch,
                    "lr": lr, "wd": wd, "L_train": L_train},
            "sec": round(time.time() - t0, 1), "curve": curve}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="reverse")
    p.add_argument("--pe", default="nope")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--nlayer", type=int, default=4)
    p.add_argument("--L_train", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    el = sorted({a.L_train, a.L_train + 2, a.L_train + 4, 2 * a.L_train})
    r = train_run(a.task, a.pe, a.seed, a.steps, L_train=a.L_train, eval_lens=el,
                  d=a.d, nlayer=a.nlayer)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    print(f"\nsec={r['sec']}  final={r['curve'][-1]}")
