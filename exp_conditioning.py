import sys
import time

import numpy as np
import torch
import torch.nn as nn

from identifiability import (SIGNAL_VAR, conditioning, log_uniform_precision, rbf)

# 先验：ell 与 sigma 都是连续 log-uniform。ell 的范围横跨「远比点距密」到「比窗口还宽」，
# 这样同一个模型内部就能扫出条件数的整条曲线，不需要为每个格子单独训一个模型。
ELL_LO, ELL_HI = 0.02, 2.0
SIG_LO, SIG_HI = 0.02, 0.5
X_HALF = 1.0
N_CTX, N_QUERY = 24, 16
N_POINTS = N_CTX + N_QUERY
D_MODEL, N_HEAD, N_LAYER = 128, 4, 6
CKPT = "pfn_cond.pt"
PRIOR_PREC = np.array([log_uniform_precision(ELL_LO, ELL_HI),
                       log_uniform_precision(SIG_LO, SIG_HI)])


class PFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_enc = nn.Linear(1, D_MODEL)
        self.y_enc = nn.Linear(1, D_MODEL)
        self.q_tok = nn.Parameter(torch.randn(D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, 4 * D_MODEL, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, N_LAYER)
        self.head = nn.Linear(D_MODEL, 2)

    def forward(self, x, y, n_ctx):
        S = x.shape[1]
        tok = self.x_enc(x.unsqueeze(-1))
        tok[:, :n_ctx] = tok[:, :n_ctx] + self.y_enc(y[:, :n_ctx].unsqueeze(-1))
        tok[:, n_ctx:] = tok[:, n_ctx:] + self.q_tok
        is_q = torch.zeros(S, dtype=torch.bool, device=x.device)
        is_q[n_ctx:] = True
        mask = is_q[None, :].expand(S, S) & ~torch.eye(S, dtype=torch.bool, device=x.device)
        out = self.enc(tok, mask=mask)
        h = self.head(out[:, n_ctx:])
        return h[..., 0], h[..., 1].clamp(-8, 3)


def sample_gp(rng, x, ell):
    """用特征分解采样。ell 接近窗口宽度时 K 极度病态，Cholesky 加抖动不可靠。"""
    K = SIGNAL_VAR * rbf(x, x, ell)
    w, V = np.linalg.eigh(K)
    return V @ (np.sqrt(np.maximum(w, 0.0)) * rng.standard_normal(len(x)))


# 对内间距取成最小的 ell。再小就要求 x 编码器分辨极细的间隔，
# 那是架构分辨率的限制，会冒充几何效应。
PAIR_GAP = ELL_LO
DESIGNS = ("uniform", "paired")


def draw_design(rng, n, kind):
    """上下文点的位置。两种设计的点数完全相同，只有位置的安排不同。

    uniform 把点铺满窗口；paired 把点两两紧邻放置，对内间距远小于最小的长度尺度。
    成对的差值直接读出局部平滑度，因此能把 ell 与 sigma 分开。
    """
    if kind == "uniform":
        return rng.uniform(-X_HALF, X_HALF, n)
    centers = rng.uniform(-X_HALF + PAIR_GAP, X_HALF - PAIR_GAP, (n + 1) // 2)
    x = np.stack([centers, centers + PAIR_GAP], axis=1).ravel()
    return x[:n]


def draw_latent(rng):
    ell = np.exp(rng.uniform(np.log(ELL_LO), np.log(ELL_HI)))
    sigma = np.exp(rng.uniform(np.log(SIG_LO), np.log(SIG_HI)))
    return ell, sigma


def make_batch(rng, bs, n_ctx):
    """上下文设计逐任务随机，两种设计都在训练分布内，评估时才能公平比较。"""
    xs, ys = np.zeros((bs, N_POINTS)), np.zeros((bs, N_POINTS))
    for b in range(bs):
        ell, sigma = draw_latent(rng)
        design = DESIGNS[int(rng.integers(len(DESIGNS)))]
        x = np.concatenate([draw_design(rng, n_ctx, design),
                            rng.uniform(-X_HALF, X_HALF, N_POINTS - n_ctx)])
        xs[b] = x
        ys[b] = sample_gp(rng, x, ell) + sigma * rng.standard_normal(N_POINTS)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def train(steps, bs=48, lr=3e-4, ckpt=CKPT):
    model = PFN()
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


def cell_geometry(ell, sigma, n_ctx, design="uniform", n_seeds=24, seed=0):
    """一个格子上的解析几何量，对上下文与查询位置取平均。

    kappa 是本文的预测量。trace_ratio、trF、trG 是三个竞争预测量，
    分别对应「各方向信息缺口相同」「只看可辨识性」「只看输出敏感度」三种失败模型。
    """
    rng = np.random.default_rng(seed)
    out = {"kappa": [], "trace_ratio": [], "trF": [], "trG": [], "vec": []}
    for _ in range(n_seeds):
        xc = draw_design(rng, n_ctx, design)
        xq = rng.uniform(-X_HALF, X_HALF, N_QUERY)
        vals, vs, F, G = conditioning(xc, xq, ell, sigma, PRIOR_PREC)
        out["kappa"].append(vals[0])
        out["trace_ratio"].append(vals.sum())
        out["trF"].append(np.trace(F))
        out["trG"].append(np.trace(G))
        out["vec"].append(vs[:, 0] * np.sign(vs[0, 0]))
    v = np.mean(out["vec"], axis=0)
    res = {k: float(np.mean(out[k])) for k in ("kappa", "trace_ratio", "trF", "trG")}
    res["vec"] = (v / np.linalg.norm(v)).tolist()
    return res


def sweep_cells():
    ells = np.exp(np.linspace(np.log(ELL_LO), np.log(ELL_HI), 8))
    sigmas = np.array([0.05, 0.2])
    n_ctxs = [8, 16, 24]
    return [(e, s, n, d) for d in DESIGNS for n in n_ctxs for s in sigmas for e in ells]


def geometry_table():
    """训练前就能算出来的条件数表。"""
    print("    A = F + Lambda 是后验精度，kappa = lam_max(A^-1 G)，trace_ratio = tr(A^-1 G)\n")
    print(f"    {'design':>9}{'n_ctx':>6}{'ell':>8}{'sigma':>7}{'kappa':>9}{'trace_ratio':>13}"
          f"{'tr F':>10}{'tr G':>9}{'峰值方向 (log ell, log sigma)':>30}")
    rows, last = [], None
    for ell, sigma, n, design in sweep_cells():
        g = cell_geometry(ell, sigma, n, design)
        rows.append((ell, sigma, n, design, g))
        if last is not None and (n, design) != last:
            print()
        last = (n, design)
        v = g["vec"]
        print(f"    {design:>9}{n:>6}{ell:>8.3f}{sigma:>7.2f}{g['kappa']:>9.3f}"
              f"{g['trace_ratio']:>13.3f}{g['trF']:>10.2f}{g['trG']:>9.3f}"
              f"{f'({v[0]:+.2f}, {v[1]:+.2f})':>30}")

    ks = np.array([g["kappa"] for *_, g in rows])
    print(f"\n    kappa 动态范围 {ks.min():.3f} – {ks.max():.3f}（{ks.max() / ks.min():.1f} 倍）")
    for design in DESIGNS:
        sub = np.array([g["kappa"] for *_, d, g in rows if d == design])
        trf = np.array([g["trF"] for *_, d, g in rows if d == design])
        print(f"    {design:>9}：kappa 中位数 {np.median(sub):.3f}，tr F 中位数 {np.median(trf):.2f}")


if __name__ == "__main__":
    if "--geometry-only" in sys.argv:
        geometry_table()
    else:
        steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
        ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
        print(f"  训练 PFN（{sum(p.numel() for p in PFN().parameters()) / 1e6:.2f}M 参数），"
              f"{steps} 步，先验 ell ∈ [{ELL_LO}, {ELL_HI}]、sigma ∈ [{SIG_LO}, {SIG_HI}]",
              flush=True)
        train(steps, ckpt=ckpt)
        print(f"  已存到 {ckpt}")
