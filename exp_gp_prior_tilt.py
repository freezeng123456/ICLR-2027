"""验证实验：PFN 算的是「自己先验下的后验」，而上下文样本能把这个先验掰过去。

之所以用高斯过程当先验，是因为它的正确答案有解析解——
给定数据，「先验 P 下的最优预测」可以精确算出来，所以能逐点核对 PFN 到底错在哪。

流程：
  1. 定义先验 P = 长度尺度 ell 取自 [0.5, 2.0] 的 GP（长度尺度大 = 函数平缓）
  2. 从 P 采样几十万个任务，训练一个小 PFN
  3. 核对：PFN 的预测是否等于 P 下的解析后验（验证「PFN = 先验 P 的化身」）
  4. 错配：拿 ell=0.15 的剧烈波动函数（远在 P 的支撑之外）喂给它，量化偏差
  5. 掰：在远离查询点的位置加入 anchor 样本——它们对查询点的局部函数值几乎不含信息，
     只携带「这个函数有多波动」这一全局信息。看 PFN 的预测是否朝解析解预测的方向移动。

第 5 步是关键。anchor 放得远，任何基于近邻检索的方法都会把它们当成无关样本丢掉，
所以如果 PFN 的预测确实被它们改变了，那就证明起作用的是「先验被掰动」而不是「多了近邻」。
"""

import time

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

NOISE = 0.1
# 先验 P 支撑的长度尺度。范围要够宽，否则似然会顶在网格边界上，anchor 就没有掰动的余地。
ELL_GRID_P = np.exp(np.linspace(np.log(0.3), np.log(3.0), 12))
D_MODEL, N_HEAD, N_LAYER = 128, 4, 6
N_POINTS, N_CTX = 40, 28


# ---------------------------------------------------------------- 高斯过程解析部分

def rbf(a, b, ell):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)


def gp_posterior(xc, yc, xq, ell):
    """给定长度尺度，返回查询点的后验均值与方差（解析解）。"""
    K = rbf(xc, xc, ell) + NOISE ** 2 * np.eye(len(xc))
    Ks = rbf(xc, xq, ell)
    alpha = np.linalg.solve(K, yc)
    mean = Ks.T @ alpha
    var = 1.0 + NOISE ** 2 - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
    return mean, np.maximum(var, 1e-6)


def log_marginal(xc, yc, ell):
    """边际似然：这批数据在该长度尺度下有多可信。"""
    K = rbf(xc, xc, ell) + NOISE ** 2 * np.eye(len(xc))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L, yc)
    return -0.5 * a @ a - np.log(np.diag(L)).sum() - 0.5 * len(xc) * np.log(2 * np.pi)


def mixture_posterior(xc, yc, xq, ell_grid):
    """先验 P 下的真实最优预测：对长度尺度的后验做加权混合。这就是 PFN 应该给出的答案。"""
    lml = np.array([log_marginal(xc, yc, e) for e in ell_grid])
    w = np.exp(lml - lml.max())
    w /= w.sum()
    means, varis = zip(*[gp_posterior(xc, yc, xq, e) for e in ell_grid])
    means, varis = np.array(means), np.array(varis)
    mu = w @ means
    var = w @ (varis + means ** 2) - mu ** 2
    return mu, var, w


def sample_gp(rng, x, ell):
    K = rbf(x, x, ell) + 1e-6 * np.eye(len(x))
    return np.linalg.cholesky(K) @ rng.standard_normal(len(x))


# ---------------------------------------------------------------- PFN 模型

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
        """x,y: (B,S)。前 n_ctx 个是上下文（带标签），其余是查询点（标签未知）。"""
        B, S = x.shape
        tok = self.x_enc(x.unsqueeze(-1))
        tok[:, :n_ctx] = tok[:, :n_ctx] + self.y_enc(y[:, :n_ctx].unsqueeze(-1))
        tok[:, n_ctx:] = tok[:, n_ctx:] + self.q_tok
        # 查询点不可被他人注意到（互不干扰），只能被自己看到
        is_q = torch.zeros(S, dtype=torch.bool, device=x.device)
        is_q[n_ctx:] = True
        mask = is_q[None, :].expand(S, S) & ~torch.eye(S, dtype=torch.bool, device=x.device)
        out = self.enc(tok, mask=mask)
        h = self.head(out[:, n_ctx:])
        return h[..., 0], h[..., 1].clamp(-6, 3)  # 均值, log 方差


def make_batch(rng, bs, n_points=N_POINTS):
    xs, ys = np.zeros((bs, n_points)), np.zeros((bs, n_points))
    for b in range(bs):
        ell = rng.choice(ELL_GRID_P)
        x = rng.uniform(-3, 3, n_points)
        xs[b], ys[b] = x, sample_gp(rng, x, ell) + NOISE * rng.standard_normal(n_points)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def train(steps, bs=48, lr=3e-4):
    model = PFN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    t0, losses = time.time(), []
    for step in range(steps):
        # 上下文长度随机，否则模型只在固定长度下可用，评估时换长度会混淆结论
        n_ctx = int(rng.integers(4, N_POINTS - 8))
        x, y = make_batch(rng, bs)
        mu, logv = model(x, y, n_ctx)
        tgt = y[:, n_ctx:]
        loss = (0.5 * (logv + (tgt - mu) ** 2 / logv.exp())).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % max(1, steps // 30) == 0:
            print(f"  step {step + 1:6d}/{steps}  loss {np.mean(losses[-200:]):7.4f}"
                  f"   已用 {time.time() - t0:5.0f}s", flush=True)
            torch.save(model.state_dict(), "pfn_gp.pt")
    return model


if __name__ == "__main__":
    import sys

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"训练一个小 PFN（{sum(p.numel() for p in PFN().parameters()) / 1e6:.2f}M 参数），"
          f"{steps} 步，先验 P = 长度尺度取自 [0.5, 2.0] 的 GP\n")
    model = train(steps)
    torch.save(model.state_dict(), "pfn_gp.pt")
    print("\n模型已存到 pfn_gp.pt")
