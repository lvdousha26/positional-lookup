"""analyze_padded.py -- re-score saved checkpoints at token level.

Sequence exact match is a conjunction over L tokens, so a model that gets 9 of 10
tokens right scores 0.0 -- indistinguishable from a model that has collapsed
completely. Before claiming a "cliff", we need the token-level curve and the
per-output-position breakdown. Both are computed here from the checkpoints the
training runs saved, so no retraining is required.

    python analyze_padded.py
"""
import glob
import json
import os

import numpy as np
import torch

from arlib import PES, TinyGPT
from padded import EOS, K, VOCAB_P, make_padded

LENS = (8, 10, 12, 16)
OUT = "../results/padded_eval"
N = 512


@torch.no_grad()
def score(model, task, L, n=N, seed=777, mask_eos=False):
    """Oracle-length evaluation (Newman et al. 2020): forbid EOS so the model
    cannot terminate early, and score what it computes."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed + L)
    inp, _, _, y = make_padded(task, L, n, g)
    logits = model(inp.to(dev))[:, K:K + L]
    if mask_eos:
        logits = logits.clone()
        logits[..., EOS] = -1e9
    pred = logits.argmax(-1).cpu()
    hit = (pred == y)                     # [n, L]
    per_pos = hit.float().mean(0)         # accuracy of output token k
    seq = hit.all(1).float().mean().item()
    # where does the model want to stop? first position predicted as EOS
    raw = model(inp.to(dev))[:, K:K + L].argmax(-1).cpu()
    stop = raw.eq(EOS).float().argmax(1).float()
    stop[~raw.eq(EOS).any(1)] = L
    return {"seq": round(seq, 4), "tok": round(hit.float().mean().item(), 4),
            "per_pos": [round(v, 4) for v in per_pos.tolist()],
            "eos": [round(v, 3) for v in raw.eq(EOS).float().mean(0).tolist()],
            "stop": round(stop.mean().item(), 2)}


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob("../results/ckpt/*.pt"))
    files = [f for f in files
             if not os.path.basename(f).startswith(("ml_", "rows_", "ne_"))]
    print(f"{len(files)} checkpoints")

    for f in files:
        name = os.path.basename(f)[:-3]
        rest, seed = name.rsplit("_s", 1)
        # `qk_abs` contains an underscore, so match against the known names
        for cand in sorted(PES, key=len, reverse=True):
            if rest.endswith("_" + cand):
                task, pe = rest[:-len(cand) - 1], cand
                break
        else:
            raise ValueError(name)
        blob = torch.load(f, map_location="cuda" if torch.cuda.is_available() else "cpu")
        cfg = blob["cfg"]
        model = TinyGPT(d=cfg["d"], nlayer=cfg["nlayer"], nhead=cfg["nhead"],
                        pe=cfg["pe"], vocab=VOCAB_P).to(
                            "cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(blob["state"])
        model.eval()

        res = {L: score(model, task, L) for L in LENS}
        orc = {L: score(model, task, L, mask_eos=True) for L in LENS}
        for L in LENS:
            res[L]["tok_oracle"] = orc[L]["tok"]
        json.dump({"task": task, "pe": pe, "seed": seed,
                   "L": {str(k): v for k, v in res.items()}},
                  open(os.path.join(OUT, name + ".json"), "w"), indent=1)
        seqs = " ".join(
            f"L{L}:{res[L]['seq']:.2f}/{res[L]['tok']:.2f}->{res[L]['tok_oracle']:.2f}"
            f"@stop{res[L]['stop']:.0f}" for L in LENS)
        print(f"{name:22s} {seqs}", flush=True)


if __name__ == "__main__":
    main()
