"""bench.py -- how does batch size change throughput on this GPU?"""
import time

import torch
import torch.nn.functional as F

from arlib import TinyGPT, make_data

torch.backends.cuda.matmul.allow_tf32 = True

for batch in (128, 512, 2048, 4096):
    model = TinyGPT(d=256, nlayer=4, nhead=8, pe="nope").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(0)

    def step():
        inp, tgt, mask, _ = make_data("reverse", 8, batch, g)
        inp, tgt, mask = inp.cuda(), tgt.cuda(), mask.cuda()
        loss = F.cross_entropy(model(inp)[mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(10):
        step()
    torch.cuda.synchronize()
    t, N = time.time(), 50
    for _ in range(N):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / N
    # time to see 256k samples, the budget of a 2000x128 run
    print(f"batch {batch:5d}  {dt*1000:6.1f} ms/step  {batch/dt:8,.0f} samples/s  "
          f"-> {256_000*batch/(batch/dt)/1000:5.1f}s per 256k samples")
