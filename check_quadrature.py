import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import exp_conditioning as gp
import exp_jump as jump
from eval_conditioning import pfn_predict, quad_grids as gp_grids
from eval_jump import quad_grids as jump_grids
from identifiability import gauss_kl, mixture_posterior as gp_posterior
from prior_jump import mixture_posterior as jump_posterior, sample_task


def midpoint_grid(lo, hi, n):
    return np.exp(np.log(lo) + (np.arange(n) + 0.5) * (np.log(hi) - np.log(lo)) / n)


def main():
    parser = argparse.ArgumentParser(description="分层检查后验求积对主要 gap 指标的影响")
    parser.add_argument("--prior", choices=["gp", "jump"], required=True)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error("tasks 必须为正数")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    family = gp if args.prior == "gp" else jump
    posterior = gp_posterior if args.prior == "gp" else jump_posterior
    lo, hi = (gp.ELL_LO, gp.ELL_HI) if args.prior == "gp" else (jump.RATE_LO, jump.RATE_HI)
    coarse = gp_grids() if args.prior == "gp" else jump_grids()
    sizes = (40, 24) if args.prior == "gp" else (28, 18)
    fine = (midpoint_grid(lo, hi, sizes[0]), midpoint_grid(family.SIG_LO, family.SIG_HI, sizes[1]))
    model, _ = gp.load_pfn("pfn_cond_40k.pt" if args.prior == "gp" else "pfn_jump_40k.pt")
    rows = []
    for latent, sigma, n, design in itertools.product((lo, hi), (0.02, 0.05, 0.2), (8, 24), family.DESIGNS):
        rng = np.random.default_rng(20260910)
        distances, gap_changes, reference_gaps = [], [], []
        for _ in range(args.tasks):
            xc = family.draw_design(rng, n, design)
            xq = rng.uniform(-1, 1, 16)
            x = np.concatenate([xc, xq])
            y = gp.sample_gp(rng, x, latent) + sigma * rng.standard_normal(len(x)) \
                if args.prior == "gp" else sample_task(rng, x, latent, sigma)
            a = posterior(xc, y[:n], xq, *coarse)
            b = posterior(xc, y[:n], xq, *fine)
            mu, var = pfn_predict(model, xc, y[:n], xq)
            gap_a, gap_b = gauss_kl(a[0], a[1], mu, var).mean(), gauss_kl(b[0], b[1], mu, var).mean()
            distances.append(float(gauss_kl(b[0], b[1], a[0], a[1]).mean()))
            gap_changes.append(float(abs(gap_a - gap_b)))
            reference_gaps.append(float(gap_b))
        rows.append({"latent": latent, "sigma": sigma, "n_context": n, "design": design,
                     "posterior_gaussian_kl": float(np.mean(distances)),
                     "gap_absolute_change": float(np.mean(gap_changes)),
                     "fine_gap": float(np.mean(reference_gaps)),
                     "relative_gap_change": float(np.mean(gap_changes) / np.mean(reference_gaps))})
        print(f"已校验 {len(rows)}/24 个条件", flush=True)
    payload = {"prior": args.prior, "tasks_per_cell": args.tasks, "seed": 20260910,
               "coarse_shape": [len(g) for g in coarse], "fine_midpoint_shape": list(sizes),
               "rows": rows, "limitation": "两种求积规则的差异检查；尚未提供连续先验误差上界"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"max_relative_gap_change": max(r["relative_gap_change"] for r in rows),
                      "max_posterior_gaussian_kl": max(r["posterior_gaussian_kl"] for r in rows)}))


if __name__ == "__main__":
    main()
