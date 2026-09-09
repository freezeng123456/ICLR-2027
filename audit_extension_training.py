import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
import torch

from audit_extension_results import sha256, true_parameters
from learned_sbi import PosteriorMDN


def audit(root, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(944317)
    n = 100000
    theta = rng.normal(size=n)
    sigma = rng.uniform(0.45, 1.2, size=n)
    offset = rng.uniform(0.3, 1.3, size=n)
    sign = 2 * rng.integers(0, 2, size=n) - 1
    context = np.column_stack((theta + sign * offset + sigma * rng.normal(size=n), sigma, offset))
    np.savez_compressed(output / "independent_validation.npz", theta=theta, context=context)
    rows = []
    for seed in range(5):
        directory = root / "training" / f"training_{seed}"
        assert (directory / "done").is_file()
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        history = json.loads((directory / "training.json").read_text())
        assert config["seed"] == seed and config["updates"] == 40000 and config["batch"] == 2048
        assert summary["updates"] == 40000 and summary["simulated_training_pairs"] == 81920000
        assert [item["step"] for item in history] == list(range(0, 40001, 2000))
        assert np.isfinite([item["validation_nll"] for item in history]).all()
        initial = torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)
        final = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(value).all() for value in final.values())
        delta = sum((final[key] - initial[key]).double().square().sum().item() for key in initial) ** 0.5
        assert delta > 0 and abs(delta - summary["parameter_delta_l2"]) < 1e-10
        assert sha256(directory / "initial.pt") == summary["initial_sha256"]
        assert sha256(directory / "final.pt") == summary["final_sha256"]
        model = PosteriorMDN()
        model.load_state_dict(final)
        model.eval()
        nlls, oracle_nlls = [], []
        with torch.no_grad():
            for start in range(0, n, 2048):
                batch = context[start:start + 2048]
                variance, means, log_weights = model(torch.tensor(batch, dtype=torch.float32))
                values = theta[start:start + 2048]
                component = norm.logpdf(values[:, None], means.numpy(), np.sqrt(variance.numpy())[:, None]) + log_weights.numpy()
                nlls.extend((-logsumexp(component, axis=-1)).tolist())
                v, m, w = true_parameters(batch)
                oracle_nlls.extend((-logsumexp(norm.logpdf(values[:, None], m, np.sqrt(v)[:, None]) + np.log(w), axis=-1)).tolist())
        differences = np.array(nlls) - oracle_nlls
        row = {"training_seed": seed, "updates": summary["updates"], "training_pairs": summary["simulated_training_pairs"],
               "parameter_count": sum(value.numel() for value in final.values()), "parameter_delta_l2": delta, "final_sha256": summary["final_sha256"],
               "recorded_initial_nll": history[0]["validation_nll"], "recorded_final_nll": history[-1]["validation_nll"],
               "independent_nll": float(np.mean(nlls)), "independent_oracle_nll": float(np.mean(oracle_nlls)),
               "independent_excess_nll": float(differences.mean()), "independent_excess_nll_se": float(differences.std(ddof=1) / np.sqrt(n)),
               "finite_parameters": True, "status": "passed"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"status": "passed", "verified_training_runs": len(rows), "independent_validation_pairs": n, "validation_rng_seed": 944317, "rows": rows}
    (output / "training_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
