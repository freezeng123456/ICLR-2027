import sys
import time

import numpy as np
import torch
import torch.nn as nn

from exp_conditioning import PFN, load_pfn
from prior_jump import sample_task

# 跳变过程先验上的 PFN。架构与高斯过程那一套完全相同，所以两边可以直接对比。
#
# 两个隐变量与高斯过程一一对应：rate 是结构那一维（速率高等于长度尺度小），
# sigma 是噪声那一维。范围取成让「结构远比点距细」到「整段窗口几乎不跳变」都覆盖到。

RATE_LO, RATE_HI = 0.5, 50.0
SIG_LO, SIG_HI = 0.02, 0.5
X_HALF = 1.0
N_CTX, N_QUERY = 24, 16
N_POINTS = N_CTX + N_QUERY
PAIR_GAP = 1.0 / RATE_HI  # 对内间距取成最细的结构尺度，与高斯过程那一套同一条约定
DESIGNS = ("uniform", "paired")
CKPT = "pfn_jump.pt"


def draw_design(rng, n, kind):
    if kind == "uniform":
        return rng.uniform(-X_HALF, X_HALF, n)
    centers = rng.uniform(-X_HALF + PAIR_GAP, X_HALF - PAIR_GAP, (n + 1) // 2)
    x = np.stack([centers, centers + PAIR_GAP], axis=1).ravel()
    return x[:n]


def draw_latent(rng):
    rate = np.exp(rng.uniform(np.log(RATE_LO), np.log(RATE_HI)))
    sigma = np.exp(rng.uniform(np.log(SIG_LO), np.log(SIG_HI)))
    return rate, sigma


def sweep_cells():
    rates = np.exp(np.linspace(np.log(RATE_LO), np.log(RATE_HI), 8))
    sigmas = np.array([0.05, 0.2])
    return [(r, s, n, d) for d in DESIGNS for n in (8, 16, 24)
            for s in sigmas for r in rates]


def make_batch(rng, bs, n_ctx):
    xs, ys = np.zeros((bs, N_POINTS)), np.zeros((bs, N_POINTS))
    for b in range(bs):
        rate, sigma = draw_latent(rng)
        design = DESIGNS[int(rng.integers(len(DESIGNS)))]
        x = np.concatenate([draw_design(rng, n_ctx, design),
                            rng.uniform(-X_HALF, X_HALF, N_POINTS - n_ctx)])
        xs[b] = x
        ys[b] = sample_task(rng, x, rate, sigma)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def train(steps, bs=48, lr=3e-4, ckpt=CKPT, d_model=128):
    model = PFN(d_model, n_head=max(1, d_model // 32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    t0, losses = time.time(), []
    for step in range(steps):
        n_ctx = int(rng.integers(6, N_POINTS - N_QUERY + 1))
        x, y = make_batch(rng, bs, n_ctx)
        mu, logv = model(x, y, n_ctx)
        tgt = y[:, n_ctx:]
        loss = (0.5 * (logv + (tgt - mu) ** 2 / logv.exp())).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"    {step + 1}/{steps}  loss {np.mean(losses[-400:]):8.4f}"
                  f"  {time.time() - t0:6.0f}s", flush=True)
            torch.save(model.state_dict(), ckpt)
    torch.save(model.state_dict(), ckpt)
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
    d_model = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    n_par = sum(p.numel() for p in PFN(d_model, max(1, d_model // 32)).parameters())
    print(f"  训练 PFN（宽度 {d_model}，{n_par / 1e6:.2f}M 参数），{steps} 步，"
          f"跳变过程先验 rate ∈ [{RATE_LO}, {RATE_HI}]、sigma ∈ [{SIG_LO}, {SIG_HI}]",
          flush=True)
    train(steps, ckpt=ckpt, d_model=d_model)
    print(f"  已存到 {ckpt}")
