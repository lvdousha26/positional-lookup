"""probe_rows.py -- how many rows does the table have?

The per-position analysis of the fixed-length protocol shows a sharp signature:
with any explicit position code the model emits exactly L_tr correct output
tokens and then something else. If the table really has L_tr rows, then moving
the training length to 12 must move the boundary to 12, at every test length.

    python probe_rows.py --task copy --pe sin --L_train 12
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from arlib import TinyGPT
from padded import K, VOCAB_P, evaluate_padded, make_padded

LENS = (10, 12, 14, 16, 18)


@torch.no_grad()
def per_pos(model, task, L, n=512, seed=777):
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed + L)
    inp, _, _, y = make_padded(task, L, n, g)
    pred = model(inp.to(dev))[:, K:K + L].argmax(-1).cpu()
    hit = (pred == y).float()
    return [round(v, 3) for v in hit.mean(0).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="copy")
    ap.add_argument("--pe", default="sin")
    ap.add_argument("--L_train", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="../results/rows")
    a = ap.parse_args()

    import padded
    ck = f"../results/ckpt/rows_{a.task}_{a.pe}_L{a.L_train}_s{a.seed}.pt"
    out = os.path.join(a.outdir, f"{a.task}_{a.pe}_L{a.L_train}_s{a.seed}.json")
    os.makedirs(a.outdir, exist_ok=True)
    if not os.path.exists(ck):
        padded.train(a.task, a.pe, a.seed, steps=a.steps, L_train=a.L_train,
                     eval_lens=(a.L_train,), ckpt=ck, verbose=False)

    blob = torch.load(ck, map_location="cuda" if torch.cuda.is_available() else "cpu")
    cfg = blob["cfg"]
    model = TinyGPT(d=cfg["d"], nlayer=cfg["nlayer"], nhead=cfg["nhead"],
                    pe=cfg["pe"], vocab=VOCAB_P).to(
                        "cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(blob["state"])
    model.eval()

    res = {}
    for L in LENS:
        pp = per_pos(model, a.task, L)
        res[str(L)] = {"per_pos": pp, "seq": round(evaluate_padded(model, a.task, L), 4)}
        # length of the correct prefix
        pref = 0
        for v in pp:
            if v > 0.9:
                pref += 1
            else:
                break
        res[str(L)]["prefix"] = pref
        print(f"L={L:2d} seq={res[str(L)]['seq']:.2f} prefix={pref:2d}  "
              + "".join("#" if v > 0.9 else ("+" if v > 0.4 else ".") for v in pp),
              flush=True)

    json.dump({"task": a.task, "pe": a.pe, "L_train": a.L_train,
               "seed": a.seed, "L": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
