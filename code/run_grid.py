"""run_grid.py -- full factorial sweep: task x positional-encoding x seed.

Each run is written to its own JSON so the grid is resumable. Runs are
independent, so we fan them out over worker processes; each worker owns its own
CUDA context (the models are a few MB, the GPU has plenty of headroom).
"""
import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from arlib import PES, TASKS
from pilot import EVAL_LENS, train_run


def run_one(cfg):
    """Worker entry point: train one configuration and persist it."""
    if os.path.exists(cfg["path"]) and not cfg["force"]:
        return f"skip  {cfg['name']}"
    t0 = time.time()
    r = train_run(cfg["task"], cfg["pe"], cfg["seed"], steps=cfg["steps"],
                  L_train=cfg["L_train"], eval_lens=EVAL_LENS, d=cfg["d"],
                  nlayer=cfg["nlayer"], nhead=cfg["nhead"], batch=cfg["batch"],
                  lr=cfg["lr"], verbose=False)
    with open(cfg["path"], "w") as fh:
        json.dump(r, fh, indent=1)
    f = r["curve"][-1]
    return (f"done  {cfg['name']:34s} in={f['L%d' % cfg['L_train']]:.3f} "
            f"ext={f['L20']:.3f}  {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--pes", default=",".join(PES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--L_train", type=int, default=8)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--nlayer", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--outdir", default="../results/grid")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    tasks, pes = a.tasks.split(","), a.pes.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]

    jobs = []
    for task, pe, seed in itertools.product(tasks, pes, seeds):
        name = (f"{a.tag}{task}__{pe}__s{seed}"
                + (f"__L{a.L_train}" if a.L_train != 8 else ""))
        jobs.append(dict(name=name, task=task, pe=pe, seed=seed,
                         path=os.path.join(a.outdir, name + ".json"),
                         steps=a.steps, L_train=a.L_train, d=a.d,
                         nlayer=a.nlayer, nhead=a.nhead, batch=a.batch,
                         lr=a.lr, force=a.force))

    print(f"{len(jobs)} jobs, {a.workers} worker(s)", flush=True)
    t0 = time.time()
    if a.workers <= 1:
        for cfg in jobs:
            print(run_one(cfg), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_one, cfg) for cfg in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(jobs)}] {fut.result()}", flush=True)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
