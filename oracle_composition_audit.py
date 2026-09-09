import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp


def schedule(t, shift=0.0, limit=15.0):
    lower = 2 / np.pi * np.arctan(np.exp(shift - limit / 2))
    upper = 2 / np.pi * np.arctan(np.exp(shift + limit / 2))
    angle = np.pi / 2 * (lower + (upper - lower) * t)
    log_snr = -2 * np.log(np.tan(angle)) + 2 * shift
    alpha2 = 1 / (1 + np.exp(-log_snr))
    beta = (1 - alpha2) * 2 * np.pi * (upper - lower) / np.sin(2 * angle)
    return np.sqrt(alpha2), 1 - alpha2, beta


def coefficients(t, n, block, noise, damping, shift, exact=False, limit=15.0):
    alpha, sigma2, beta = schedule(t, shift, limit)
    variance = noise / (noise + n)
    mean = n / (noise + n)
    if exact:
        vt = alpha**2 * variance + sigma2
        return 1 / vt, alpha * mean / vt, beta
    groups = n // block
    local_variance = noise / (noise + block)
    local_mean = block / (noise + block)
    vt = alpha**2 * local_variance + sigma2
    d = np.exp(t * np.log(damping))
    return d * (groups / vt - (groups - 1) * (1 - t)), d * groups * alpha * local_mean / vt, beta


def integrate(n, block, noise, damping, shift, dynamics, exact=False,
              method="DOP853", tolerance=1e-10, limit=15.0):
    assert n % block == 0
    target_mean = n / (noise + n)
    target_variance = noise / (noise + n)
    alpha, sigma2, _ = schedule(1, shift, limit)
    initial = [alpha * target_mean, alpha**2 * target_variance + sigma2] if exact else [0, block / (n * damping)]
    if dynamics == "ode":
        initial[1] = np.log(initial[1])

    def rhs(t, state):
        precision, linear, beta = coefficients(t, n, block, noise, damping, shift, exact, limit)
        if dynamics == "ode":
            return [0.5 * beta * ((precision - 1) * state[0] - linear),
                    beta * (precision - 1)]
        return [beta * ((precision - 0.5) * state[0] - linear),
                beta * ((2 * precision - 1) * state[1] - 1)]

    solution = solve_ivp(rhs, (1, 0), initial, method=method, rtol=tolerance,
                         atol=tolerance * 0.001)
    assert solution.success, solution.message
    mean, variance = solution.y[:, -1]
    if dynamics == "ode":
        variance = np.exp(variance)
    assert np.isfinite([mean, variance]).all() and variance > 0
    kl = 0.5 * (np.log(variance / target_variance) +
                (target_variance + (mean - target_mean)**2) / variance - 1)
    alpha0, sigma20, _ = schedule(0, shift, limit)
    expected_terminal = [alpha0 * target_mean, alpha0**2 * target_variance + sigma20]
    if exact:
        np.testing.assert_allclose([mean, variance], expected_terminal, atol=2e-8, rtol=2e-8)
    return dict(n=n, block=block, noise=noise, damping=damping, shift=shift,
                dynamics=dynamics, exact=exact, method=method, tolerance=tolerance,
                snr_limit=limit, mean=float(mean), variance=float(variance),
                target_mean=target_mean, target_variance=target_variance,
                variance_ratio=float(variance / target_variance),
                mean_error_posterior_sd=float((mean-target_mean)/np.sqrt(target_variance)),
                kl_p_q=float(kl), evaluations=solution.nfev)


def official_check(root):
    sys.path.insert(0, str(root.resolve()))
    import torch
    from diffusion_model.diffusion_sde import SDE
    from diffusion_model.sampling_algorithms import euler_step, euler_maruyama_step

    torch.set_default_dtype(torch.float64)

    class OracleDynamics:
        def __init__(self, shift):
            self.sde = SDE(noise_schedule="cosine", s_shift_cosine=shift)

    maximum = 0.0
    for shift in [0.0, 2.0]:
        oracle = OracleDynamics(shift)
        for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
            time = torch.tensor(t)
            a, s2, beta = schedule(t, shift)
            official_a, official_s = oracle.sde.kernel(oracle.sde.get_snr(time))
            f, g = oracle.sde.get_f_g(t=time, x=torch.tensor(1.0))
            expected = [a, s2, -beta / 2, beta]
            actual = [official_a.item(), official_s.square().item(), f.item(), g.square().item()]
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-10)
            maximum = max(maximum, float(np.max(np.abs(np.array(actual)-expected))))
            precision, linear, _ = coefficients(t, 16, 1, 1, 1/16, shift)
            x = torch.tensor([[-0.5], [0.7]])
            score = -precision * x + linear
            h = torch.tensor(1e-5)
            actual_ode = euler_step(oracle, x, score, time, h)
            expected_ode = x - h * beta / 2 * ((precision-1)*x-linear)
            torch.testing.assert_close(actual_ode, expected_ode, atol=1e-10, rtol=1e-10)
            actual_sde = euler_maruyama_step(oracle, x, score, time, h, noise=torch.zeros_like(x))
            expected_sde = x - h * beta * ((precision-0.5)*x-linear)
            torch.testing.assert_close(actual_sde, expected_sde, atol=1e-10, rtol=1e-10)
    return dict(status="PASS", times=10, maximum_absolute_difference=maximum,
                commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                torch=torch.__version__)


def gauss_identity_check():
    maximum = 0.0
    for n, block in [(4, 1), (16, 1), (16, 4)]:
        groups = n // block
        for t in np.linspace(0.001, 0.999, 99):
            a, s2, _ = schedule(t)
            local_variance = 1 / (1 + block)
            local_mean = block / (1 + block)
            vt = a*a*local_variance+s2
            backward_prior_precision = 1 + a*a/s2
            backward_local_precision = 1/local_variance+a*a/s2
            combined = groups*backward_local_precision+(1-groups)*backward_prior_precision
            c = (groups*backward_local_precision/vt+(1-groups)*backward_prior_precision)/combined
            b = groups*backward_local_precision*a*local_mean/vt/combined
            exact_c, exact_b, _ = coefficients(t, n, block, 1, 1, 0, exact=True)
            np.testing.assert_allclose([c, b], [exact_c, exact_b], rtol=2e-10, atol=2e-10)
            maximum = max(maximum, abs(c-exact_c), abs(b-exact_b))
    return dict(status="PASS", cases=297, maximum_absolute_difference=maximum,
                scope="Linhart GAUSS algebra with exact Gaussian covariances; no learned network")


def no_information_check():
    rows = []
    for groups in [1, 4, 16]:
        def integrand(t):
            c = np.exp(-t*np.log(groups)) * (1+(groups-1)*t)
            return np.pi*np.tan(np.pi*t/2)*(c-1)
        integral, error = quad(integrand, 0, 1, epsabs=1e-11, epsrel=1e-11)
        rows.append(dict(groups=groups, variance=float(np.exp(-integral)), target_variance=1.0,
                         integral=float(integral), quadrature_error=float(error)))
    return dict(scope="Exact untruncated cosine VP; independent observations with likelihood constant in theta", rows=rows)


def minibatch_square_check():
    noise = np.array([0.5, 1.0, 2.0, 4.0])
    observations = np.array([-1.0, 0.2, 0.7, 1.3])
    local_variance = noise / (1+noise)
    local_mean = observations / (1+noise)
    a, s2, _ = schedule(0.4)
    batches = np.array(list(itertools.product(range(4), repeat=2)))
    rows = []
    for x in [-1.0, 0.0, 1.0]:
        scores = -(x-a*local_mean)/(a*a*local_variance+s2)
        prior_term = 3*x
        full = prior_term+scores.sum()
        estimates = prior_term+4*scores[batches].mean(axis=1)
        plugin = np.mean(estimates**2)
        cross = np.mean(estimates[:, None]*estimates[None, :])
        bias = 4**2/2*np.var(scores)
        np.testing.assert_allclose(plugin-full**2, bias, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cross, full**2, rtol=1e-12, atol=1e-12)
        rows.append(dict(x=x, exact_square=float(full**2), plugin_expectation=float(plugin),
                         predicted_bias=float(bias), cross_expectation=float(cross)))
    return dict(status="PASS", sampling="uniform with replacement, batch size 2, 4 Gaussian factors",
                scope="All 16 batches and 256 independent batch pairs enumerated; no sampler claim", rows=rows)


def plot_results(rows, destination):
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    x = np.linspace(-0.15, 1.9, 1800)
    target = next(r for r in rows if r["n"] == 16 and r["exact"])
    axes[0].plot(x, norm.pdf(x, target["target_mean"], np.sqrt(target["target_variance"])),
                 color="#182d43", lw=2.5, label="Exact posterior")
    for dynamics, color in [("ode", "#ba4b32"), ("sde", "#28877a")]:
        selected = [r for r in rows if r["n"] == 16 and not r["exact"] and
                    r["dynamics"] == dynamics and r["damping"] == r["block"]/r["n"]]
        singleton = next(r for r in selected if r["block"] == 1)
        axes[0].plot(x, norm.pdf(x, singleton["mean"], np.sqrt(singleton["variance"])),
                     color=color, lw=2, label=f"Composed {dynamics.upper()}, 16 blocks")
        axes[1].plot([16//r["block"] for r in selected], [r["variance_ratio"] for r in selected],
                     "o-", color=color, lw=2, label=dynamics.upper())
    axes[0].set(xlabel="Parameter", ylabel="Density", title="Perfect local scores; 16 observations")
    axes[0].legend(fontsize=8.3)
    axes[1].axhline(1, color="#182d43", linestyle="--", label="Exact ratio = 1")
    axes[1].set(xlabel="Number of blocks (same total data)", ylabel="Output variance / exact variance",
                title="Partition dependence", xticks=[1, 4, 16], ylim=(-0.04, 1.09))
    axes[1].legend(fontsize=8.5, loc="center right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.savefig(destination / "oracle-composition.png", dpi=180)
    fig.savefig(destination / "oracle-composition.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = official_check(args.official_repo)
    rows = []
    for n in [1, 4, 16]:
        for dynamics in ["ode", "sde"]:
            rows.append(integrate(n, n, 1, 1, 0, dynamics, exact=True))
            for block in sorted({1, n // 4 if n >= 4 else 1, n}):
                for damping in sorted({1.0, block/n}):
                    rows.append(integrate(n, block, 1, damping, 0, dynamics))
    convergence = []
    for dynamics in ["ode", "sde"]:
        for limit in [15.0, 25.0]:
            for method in ["DOP853", "Radau"]:
                convergence.append(integrate(16, 1, 1, 1/16, 0, dynamics,
                                              method=method, tolerance=1e-11, limit=limit))
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(status="PASS", experiment_type="analytic_oracle_no_training",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  official_check=checks, gauss_identity_check=gauss_identity_check(),
                  no_information_check=no_information_check(), minibatch_square_check=minibatch_square_check(),
                  rows=rows, convergence=convergence)
    (args.output / "verification.json").write_text(json.dumps(record, indent=2)+"\n")
    plot_results(rows, args.output)
    print(json.dumps(dict(status="PASS", cases=len(rows), gauss=record["gauss_identity_check"],
                          no_information=record["no_information_check"]), indent=2))


if __name__ == "__main__":
    main()
