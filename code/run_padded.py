"""run_padded.py -- injection-site experiment on the fixed-length protocol.

Design: hold the position *code* fixed (the same sinusoidal vector) and vary only
where it enters the network:

    site = representation : sin       (absolute, per-token)
    site = q/k            : qk_abs    (absolute, pairwise site)
    site = q/k            : rope      (relative)
    site = logits         : alibi     (relative; the only option, see below)
    site = nowhere        : nope

A purely additive bias on the attention logits is annihilated by softmax when it
is constant along the key axis, so an *effective* logit code is necessarily
pairwise -- i.e. relative. Hence the logits site cannot host an absolute code,
which is what makes `qk_abs` the decisive cell: it is the only condition with an
absolute code at a site that could in principle carry a relative one. If it
collapses like `sin`, the operative variable is absoluteness, not the site.

Usage:
    python run_padded.py --tasks copy,reverse,sort --seeds 0,1 --steps 2000
"""
import argparse
import json
import os
import sys
import time

import padded

PES = ("nope", "sin", "qk_abs", "rope", "alibi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="copy,reverse,sort")
    ap.add_argument("--pes", default=",".join(PES))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--outdir", default="../results/padded")
    ap.add_argument("--ckptdir", default="../results/ckpt")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    tasks = a.tasks.split(",")
    pes = a.pes.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(a.ckptdir, exist_ok=True)

    todo = [(t, p, s) for t in tasks for p in pes for s in seeds
            if a.force or not os.path.exists(
                os.path.join(a.outdir, f"{t}_{p}_s{s}.json"))]
    print(f"{len(todo)} runs queued", flush=True)

    for i, (task, pe, seed) in enumerate(todo, 1):
        out = os.path.join(a.outdir, f"{task}_{pe}_s{seed}.json")
        ck = os.path.join(a.ckptdir, f"{task}_{pe}_s{seed}.pt")
        t0 = time.time()
        r = padded.train(task, pe, seed, steps=a.steps, ckpt=ck, verbose=False)
        r["idx"] = i
        json.dump(r, open(out, "w"), indent=1)
        c = r["curve"][-1]
        cells = " ".join(f"L{k[1:]}={v:.3f}" for k, v in c.items() if k.startswith("L"))
        print(f"[{i:2d}/{len(todo)}] {task:8s} {pe:7s} s{seed}  {cells}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
