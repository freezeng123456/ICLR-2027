"""为什么噪声轴能掰动、平滑度轴不能？

假说：一个轴能否被上下文掰动，取决于它在**预训练先验里是否是一个会变化、
因而必须逐任务推断的隐变量**。先验里从不变化的属性，模型没有表示它的必要，
也就无从掰起。

用 2x2 设计证伪。训三个先验不同的 PFN：

    模型            长度尺度      噪声水平
    A(只有尺度变)     变化        固定
    B(只有噪声变)     固定        变化
    C(都变)          变化        变化

每个模型都测两种掰动，anchor 一律远置：
    尺度掰动：anchor 取自波动的函数 vs 平滑的函数（噪声都很低）
             读数 = 查询点**预测均值**的移动
    噪声掰动：anchor 取自同样尺度的函数，但观测噪声一低一高
             读数 = 查询点**预测标准差**的移动

预测的结果：
    模型 A  尺度掰动有效，噪声掰动无效
    模型 B  尺度掰动无效，噪声掰动有效
    模型 C  两者都有效

若这个交互出现，假说成立，并且直接解释了表格 PFN 的结果——
噪声几乎一定是 SCM 先验里的显式参数，而「函数是否平滑」不是。
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn

D_MODEL, N_HEAD, N_LAYER = 128, 4, 6
N_POINTS = 40

# 三个先验配置：(长度尺度候选, 噪声候选)
PRIORS = {
    "A(只有尺度变)": (np.exp(np.linspace(np.log(0.3), np.log(3.0), 12)), np.array([0.1])),
    "B(只有噪声变)": (np.array([1.0]), np.exp(np.linspace(np.log(0.02), np.log(0.5), 12))),
    "C(都变)": (np.exp(np.linspace(np.log(0.3), np.log(3.0), 12)),
                np.exp(np.linspace(np.log(0.02), np.log(0.5), 12))),
}


def rbf(a, b, ell):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)


def sample_gp(rng, x, ell):
    K = rbf(x, x, ell) + 1e-6 * np.eye(len(x))
    return np.linalg.cholesky(K) @ rng.standard_normal(len(x))


def gp_posterior(xc, yc, xq, ell, noise):
    K = rbf(xc, xc, ell) + noise ** 2 * np.eye(len(xc))
    Ks = rbf(xc, xq, ell)
    mean = Ks.T @ np.linalg.solve(K, yc)
    var = 1.0 + noise ** 2 - np.sum(Ks * np.linalg.solve(K, Ks), axis=0)
    return mean, np.maximum(var, 1e-6)


def log_marginal(xc, yc, ell, noise):
    K = rbf(xc, xc, ell) + noise ** 2 * np.eye(len(xc))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L, yc)
    return -0.5 * a @ a - np.log(np.diag(L)).sum() - 0.5 * len(xc) * np.log(2 * np.pi)


def mixture_posterior(xc, yc, xq, ells, noises):
    """在 (长度尺度, 噪声) 的联合网格上做边缘化，得到该先验下的精确最优预测。"""
    grid = [(e, n) for e in ells for n in noises]
    lml = np.array([log_marginal(xc, yc, e, n) for e, n in grid])
    w = np.exp(lml - lml.max())
    w /= w.sum()
    ms, vs = zip(*[gp_posterior(xc, yc, xq, e, n) for e, n in grid])
    ms, vs = np.array(ms), np.array(vs)
    mu = w @ ms
    var = w @ (vs + ms ** 2) - mu ** 2
    return mu, var


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
        is_q = torch.zeros(S, dtype=torch.bool)
        is_q[n_ctx:] = True
        mask = is_q[None, :].expand(S, S) & ~torch.eye(S, dtype=torch.bool)
        out = self.enc(tok, mask=mask)
        h = self.head(out[:, n_ctx:])
        return h[..., 0], h[..., 1].clamp(-6, 3)


def make_batch(rng, bs, ells, noises):
    xs, ys = np.zeros((bs, N_POINTS)), np.zeros((bs, N_POINTS))
    for b in range(bs):
        ell, noise = rng.choice(ells), rng.choice(noises)
        x = rng.uniform(-3, 3, N_POINTS)
        xs[b] = x
        ys[b] = sample_gp(rng, x, ell) + noise * rng.standard_normal(N_POINTS)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def train(name, ells, noises, steps, bs=48, lr=3e-4):
    model = PFN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    t0, losses = time.time(), []
    for step in range(steps):
        n_ctx = int(rng.integers(4, N_POINTS - 8))
        x, y = make_batch(rng, bs, ells, noises)
        mu, logv = model(x, y, n_ctx)
        tgt = y[:, n_ctx:]
        loss = (0.5 * (logv + (tgt - mu) ** 2 / logv.exp())).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if (step + 1) % 2500 == 0:
            print(f"    [{name}] {step + 1}/{steps}  loss {np.mean(losses[-300:]):7.4f}"
                  f"  {time.time() - t0:5.0f}s", flush=True)
    torch.save(model.state_dict(), f"pfn_{name[0]}.pt")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    for name, (ells, noises) in PRIORS.items():
        print(f"  训练 {name}：尺度候选 {len(ells)} 个，噪声候选 {len(noises)} 个", flush=True)
        train(name, ells, noises, steps)
    print("  三个模型训练完成")
