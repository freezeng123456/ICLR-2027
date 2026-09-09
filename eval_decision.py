import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import exp_conditioning as gp
import exp_jump as jump
from eval_conditioning import pfn_predict, quad_grids as gp_grids
from eval_jump import quad_grids as jump_grids
from identifiability import mixture_posterior as gp_posterior
from prior_jump import mixture_posterior as jump_posterior, sample_task
from train_repro import sha256


COVERAGES = np.array([0.1, 0.2, 0.5, 0.8, 1.0])


def selected_risk(score, loss, coverages=COVERAGES):
    score, loss = np.asarray(score), np.asarray(loss)
    if score.ndim != 1 or score.shape != loss.shape or not len(score):
        raise ValueError("score 和 loss 必须是同长度的非空向量")
    if not np.isfinite(score).all() or not np.isfinite(loss).all():
        raise ValueError("风险计算要求有限数值")
    order = np.argsort(score, kind="stable")
    count = np.maximum(1, np.ceil(coverages * len(score)).astype(int))
    return np.cumsum(loss[order])[count - 1] / count


def decision_curves(mu_p, var_p, mu_e, var_e, y):
    # 固定网络均值时，oracle 排序要同时考虑后验方差与均值误差。
    risk_p = var_e + (mu_p - mu_e) ** 2
    return {"network": selected_risk(var_p, risk_p),
            "same_mean_oracle": selected_risk(risk_p, risk_p),
            "bayes": selected_risk(var_e, var_e),
            "empirical_network": selected_risk(var_p, (y - mu_p) ** 2)}


def summarize(records, bootstrap=1000, seed=0):
    names = records[0]["predictions"].keys()
    output = {}
    for name in names:
        def curves(indices):
            fields = {k: np.concatenate([np.asarray(records[i][k]) for i in indices])
                      for k in ["mu_e", "var_e", "y"]}
            pred = {k: np.concatenate([np.asarray(records[i]["predictions"][name][k]) for i in indices])
                    for k in ["mu_p", "var_p"]}
            return decision_curves(**fields, **pred)

        point = curves(range(len(records)))
        regret = point["network"] - point["same_mean_oracle"]
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(bootstrap):
            draw = curves(rng.integers(0, len(records), len(records)))
            samples.append(draw["network"] - draw["same_mean_oracle"])
        output[name] = {**{k: v.tolist() for k, v in point.items()},
                        "selection_regret": regret.tolist(),
                        "total_regret_vs_bayes": (point["network"] - point["bayes"]).tolist(),
                        "mean_component": (point["same_mean_oracle"] - point["bayes"]).tolist(),
                        "regret_ci95": np.quantile(samples, [0.025, 0.975], axis=0).T.tolist()}
    return output


def main():
    parser = argparse.ArgumentParser(description="在先验抽样任务上测量固定预测器的选择性预测代价")
    parser.add_argument("--prior", choices=["gp", "jump"], required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--n-context", type=int, choices=[8, 16, 24], default=24)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tasks < 2 or args.bootstrap < 1:
        parser.error("tasks 至少 2，bootstrap 必须为正数")
    if args.output.exists():
        raise FileExistsError(args.output)
    if len({p.name for p in args.checkpoints}) != len(args.checkpoints):
        parser.error("检查点文件名必须互不相同")
    torch.set_num_threads(args.threads)
    family = gp if args.prior == "gp" else jump
    posterior = gp_posterior if args.prior == "gp" else jump_posterior
    grids = gp_grids() if args.prior == "gp" else jump_grids()
    models = {p.name: gp.load_pfn(p)[0] for p in args.checkpoints}
    rng = np.random.default_rng(args.seed)
    records = []
    for task in range(args.tasks):
        latent, sigma = family.draw_latent(rng)
        design = family.DESIGNS[int(rng.integers(2))]
        xc = family.draw_design(rng, args.n_context, design)
        xq = rng.uniform(-1, 1, 16)
        x = np.concatenate([xc, xq])
        y = gp.sample_gp(rng, x, latent) + sigma * rng.standard_normal(len(x)) \
            if args.prior == "gp" else sample_task(rng, x, latent, sigma)
        yc = y[:args.n_context]
        mu_e, var_e, _ = posterior(xc, yc, xq, *grids)
        predictions = {}
        for name, model in models.items():
            mu_p, var_p = pfn_predict(model, xc, yc, xq)
            predictions[name] = {"mu_p": mu_p.tolist(), "var_p": var_p.tolist()}
        records.append({"task": task, "latent": latent, "sigma": sigma, "design": design,
                        "xc": xc.tolist(), "yc": yc.tolist(), "xq": xq.tolist(),
                        "mu_e": mu_e.tolist(), "var_e": var_e.tolist(),
                        "y": y[args.n_context:].tolist(), "predictions": predictions})
        if (task + 1) % 10 == 0:
            print(f"已评估 {task + 1}/{args.tasks} 个任务", flush=True)
    repo = Path(__file__).resolve().parent
    payload = {"prior": args.prior, "tasks": args.tasks, "n_context": args.n_context,
               "seed": args.seed, "coverages": COVERAGES.tolist(), "bootstrap": args.bootstrap,
               "bootstrap_unit": "task", "coverage_scope": "pooled_across_tasks_and_queries",
               "task_distribution": "training_prior",
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
               "source_sha256": {p: sha256(repo / p) for p in
                                 ["eval_decision.py", "exp_conditioning.py", "exp_jump.py",
                                  "identifiability.py", "prior_jump.py"]},
               "checkpoint_sha256": {p.name: sha256(p) for p in args.checkpoints},
               "quadrature_shape": [len(g) for g in grids], "records": records,
               "summary": summarize(records, args.bootstrap, args.seed),
               "limitations": ["单组历史训练检查点，bootstrap 仅反映任务抽样",
                               "Bayesian 风险以数值求积后验为参照，尚需困难区域收敛检查",
                               "CI 是逐检查点逐覆盖率区间，未作多重比较修正"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
