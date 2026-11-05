"""check_tasks.py -- verify sequence alignment and loss-mask bookkeeping."""
import torch

from arlib import COUPLED, TASKS, make_data

for task in TASKS:
    L = 8
    g = torch.Generator().manual_seed(0)
    inp, tgt, mask, y = make_data(task, L, 2, g)
    print("=" * 62)
    print(f"{task}  (L-coupled={task in COUPLED})  inp={tuple(inp.shape)} "
          f"tgt={tuple(tgt.shape)} mask_sum={int(mask.sum())}")
    for b in range(2):
        print("   x   :", inp[b, :L].tolist())
        print("   y   :", y[b].tolist())
        print("   tgt :", tgt[b].tolist())
        print("   mask:", mask[b].int().tolist())
    # tgt[L:L+L] must equal y (positions that predict y_1..y_L)
    assert torch.equal(tgt[:, L:L + L], y), task
    # nothing before the SEP is supervised; everything from y_1 to EOS is
    assert mask[:, :L].sum() == 0
    assert bool(mask[:, L:].all())
print("\nOK: alignment and mask verified for all tasks.")
