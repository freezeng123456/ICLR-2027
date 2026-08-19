"""补回两个丢失的 checkpoint，并修掉导致丢失的存盘 bug。

背景：exp_why_axis.py 里的 train() 用 f"pfn_{name[0]}.pt" 存盘，只取名字首字母。
exp_c_longer.py 传的名字是 "C_long"，首字母还是 C，于是 45000 步的模型覆盖了
15000 步的。加上 exp_entangle.py 的 A144 从未提交，论文里那张排混淆的关键对比表
现在两行都无法从仓库文件重现。

这个脚本用**显式路径**存盘（不再从名字推），补回：
    pfn_Z.pt        A144：144 档长度尺度、噪声固定、15000 步
    pfn_C15000.pt   C：12x12 组合、15000 步（不覆盖现有的 45000 步版本）

要真正修掉 bug，还需要把 exp_why_axis.py 的 train() 改成接收显式路径——
见文件末尾的说明。
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn

from exp_why_axis import PFN, PRIORS, make_batch

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ELLS_144 = np.exp(np.linspace(np.log(0.3), np.log(3.0), 144))
NOISE_FIXED = np.array([0.1])
N_POINTS = 40


def train_to(path, ells, noises, steps, bs=48, lr=3e-4, tag="", seed=SEED):
    """和原 train() 相同的训练过程，两处修正：

    1. **存盘路径由调用方显式给出**，不再从名字首字母推导
       （原来 f"pfn_{name[0]}.pt" 让 "C_long" 覆盖了 "C"）。
    2. **显式设 torch 种子**。原 exp_why_axis.py 只设了 numpy 种子，
       torch 的权重初始化没设，导致训练不可复现——实测同样配置重训，
       相关系数会差 0.09 到 0.11。
    """
    torch.manual_seed(seed)
    model = PFN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)          # 与原脚本同种子，保证可比
    t0, losses = time.time(), []
    for step in range(steps):
        n_ctx = int(rng.integers(4, N_POINTS - 8))
        x, y = make_batch(rng, bs, ells, noises)
        mu, logv = model(x, y, n_ctx)
        tgt = y[:, n_ctx:]
        loss = (0.5 * (logv + (tgt - mu) ** 2 / logv.exp())).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % 2500 == 0:
            print(f"    [{tag}] {step + 1}/{steps}  loss {np.mean(losses[-300:]):7.4f}"
                  f"  {time.time() - t0:5.0f}s", flush=True)
    torch.save(model.state_dict(), path)
    print(f"    [{tag}] 已存到 {path}", flush=True)
    return model


if __name__ == "__main__":
    print(f"  用 {torch.get_num_threads()} 个线程，各训 {STEPS} 步，torch 种子 {SEED}")
    print("  用法：python3 rebuild_checkpoints.py [步数] [线程数] [种子]\n")

    print("  (1) A144：144 档长度尺度、噪声固定 0.1 —— 排混淆实验用")
    train_to("pfn_Z.pt", ELLS_144, NOISE_FIXED, STEPS, tag="A144")

    print("\n  (2) C@15000：12x12 组合 —— 主对比表里的那个 C")
    ells_c, noises_c = PRIORS["C(都变)"]
    train_to("pfn_C15000.pt", ells_c, noises_c, STEPS, tag="C15000")

    print("""
  完成。

  要在源仓库里根治这两个 bug，把 exp_why_axis.py 的 train() 改成：

      def train(name, ells, noises, steps, bs=48, lr=3e-4, path=None, seed=0):
          torch.manual_seed(seed)                                   # <- 现在缺这一行
          ...
          torch.save(model.state_dict(), path or f"pfn_{name}.pt")  # <- 用完整名字

  两处修正各自解决一个问题：
    manual_seed  ——  原来只设了 numpy 种子，权重初始化随机，训练不可复现。
                     实测同配置重训，相关系数差 0.09 到 0.11，而论文里
                     A144 vs C 的关键对比就建立在单次训练上，必须报多个种子。
    完整名字      ——  原来 name[0] 让 exp_c_longer.py 的 "C_long" 覆盖了 "C"，
                     15000 步的 checkpoint 永久丢失。
  对应地 exp_c_longer.py 改成 train("C_long", ..., path="pfn_C_long.pt")。
""")
