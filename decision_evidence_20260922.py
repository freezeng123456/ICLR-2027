from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def growth_optimal_mass(positive_mass, negative_mass):
    positive_mass = np.asarray(positive_mass, dtype=float)
    negative_mass = np.asarray(negative_mass, dtype=float)
    if np.any(positive_mass <= 0) or np.any(negative_mass <= 0):
        raise ValueError("Both signed masses must be strictly positive")
    log_ratio = np.log(positive_mass) - np.log(negative_mass)
    result = np.empty(np.broadcast_shapes(log_ratio.shape, ()), dtype=float)
    near = np.abs(log_ratio) < 1e-3
    z = log_ratio[near]
    result[near] = 0.5 + z / 12 - z**3 / 720 + z**5 / 30240
    z = log_ratio[~near]
    result[~near] = -1 / np.expm1(-z) - 1 / z
    return result


def oracle_bet(positive_mass, negative_mass, sampling_mass):
    if min(positive_mass, negative_mass) <= 0 or not 0 < sampling_mass < 1:
        raise ValueError("Masses must be positive and proposal mass in (0, 1)")
    return (
        (positive_mass - negative_mass)
        * sampling_mass
        * (1 - sampling_mass)
        / (positive_mass * negative_mass)
    )


def oracle_growth(positive_mass, negative_mass, sampling_mass):
    if min(positive_mass, negative_mass) <= 0 or not 0 < sampling_mass < 1:
        raise ValueError("Invalid mass")
    mixture = positive_mass * (1 - sampling_mass) + negative_mass * sampling_mass
    return (
        np.log(mixture)
        - sampling_mass * np.log(negative_mass)
        - (1 - sampling_mass) * np.log(positive_mass)
    )


def numerical_oracle_mass(positive_mass, negative_mass):
    solution = minimize_scalar(
        lambda mass: -oracle_growth(positive_mass, negative_mass, mass),
        bounds=(1e-6, 1 - 1e-6),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.x


@dataclass(frozen=True)
class BoundedStratumProblem:
    name: str
    positive_scale: float
    negative_scale: float
    positive_beta: tuple[float, float]
    negative_beta: tuple[float, float]

    def __post_init__(self):
        parameters = (
            self.positive_scale,
            self.negative_scale,
            *self.positive_beta,
            *self.negative_beta,
        )
        if min(parameters) <= 0:
            raise ValueError("All problem parameters must be positive")

    @property
    def integral(self):
        a, b = self.positive_beta
        c, d = self.negative_beta
        return self.positive_scale * a / (a + b) - self.negative_scale * c / (c + d)

    def sample(self, rng, positive, size):
        shape = self.positive_beta if positive else self.negative_beta
        return rng.beta(*shape, size=size)


class EvidenceState:
    def __init__(self, alpha=0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.log_positive = 0.0
        self.log_negative = 0.0
        self.samples = 0
        self.decision = 0

    def observe(self, weighted_value, positive_bet, negative_bet):
        if positive_bet < 0 or negative_bet < 0:
            raise ValueError("Bets must be nonnegative")
        plus = positive_bet * weighted_value
        minus = -negative_bet * weighted_value
        if not np.isfinite(plus + minus) or min(plus, minus) <= -1:
            raise ValueError("Bet violates nonnegative-wealth constraint")
        self.log_positive += float(np.log1p(plus))
        self.log_negative += float(np.log1p(minus))
        self.samples += 1
        boundary = np.log(2 / self.alpha)
        if self.decision == 0:
            if self.log_positive >= boundary:
                self.decision = 1
            elif self.log_negative >= boundary:
                self.decision = -1
        return self.decision


def empirical_bets(positive_values, negative_values, positive_scale, negative_scale, masses):
    masses = np.asarray(masses, dtype=float)
    if np.any(masses <= 0) or np.any(masses >= 1):
        raise ValueError("Proposal mass must have full support")
    if len(positive_values) == 0 or len(negative_values) == 0:
        raise ValueError("Each stratum needs observations")
    fractions = np.linspace(0.0, 0.99, 32)
    plus_bets = fractions[None, :] * (1 - masses[:, None]) / negative_scale
    minus_bets = fractions[None, :] * masses[:, None] / positive_scale
    # 固定64个近期观测限制优化开销；该近似只影响效率，不参与证据有效性。
    pos = positive_scale * np.asarray(positive_values[-64:])
    neg = negative_scale * np.asarray(negative_values[-64:])

    def expectation(bets, direction):
        positive_logs = np.log1p(direction * bets[:, :, None] * pos / masses[:, None, None])
        negative_logs = np.log1p(-direction * bets[:, :, None] * neg / (1 - masses[:, None, None]))
        return (
            masses[:, None] * positive_logs.mean(axis=2)
            + (1 - masses[:, None]) * negative_logs.mean(axis=2)
        )

    positive_growth = expectation(plus_bets, 1)
    negative_growth = expectation(minus_bets, -1)
    plus_indices = positive_growth.argmax(axis=1)
    minus_indices = negative_growth.argmax(axis=1)
    rows = np.arange(len(masses))
    return (
        plus_bets[rows, plus_indices],
        minus_bets[rows, minus_indices],
        positive_growth[rows, plus_indices],
        negative_growth[rows, minus_indices],
    )


def choose_design(method, positive_values, negative_values, positive_scale, negative_scale):
    if method not in {"uniform", "variance", "growth"}:
        raise ValueError(f"Unknown method: {method}")
    if method == "uniform":
        masses = np.array([0.5])
    elif method == "variance":
        # 这里最小化真实抽样变量的经验二阶矩，不使用正负均值之比代替。
        pos = positive_scale * np.sqrt(np.mean(np.square(positive_values)))
        neg = negative_scale * np.sqrt(np.mean(np.square(negative_values)))
        masses = np.array([np.clip(pos / (pos + neg), 0.02, 0.98)])
    else:
        masses = np.linspace(0.02, 0.98, 33)
    plus, minus, plus_growth, minus_growth = empirical_bets(
        positive_values, negative_values, positive_scale, negative_scale, masses
    )
    index = int(np.maximum(plus_growth, minus_growth).argmax())
    return float(masses[index]), float(plus[index]), float(minus[index])


def run_bounded_trial(problem, method, seed, max_samples=4096, alpha=0.05, batch_size=32):
    if max_samples < 4 or batch_size < 1:
        raise ValueError("Invalid sample budget")
    rng = np.random.default_rng(seed)
    positive_values = list(problem.sample(rng, True, 2))
    negative_values = list(problem.sample(rng, False, 2))
    evidence = EvidenceState(alpha)
    calls = 4
    trace = []
    while calls < max_samples and evidence.decision == 0:
        mass, plus_bet, minus_bet = choose_design(
            method, positive_values, negative_values, problem.positive_scale, problem.negative_scale
        )
        # 本批次的提议与两个赌注只使用此前已经观察到的数据。
        count = min(batch_size, max_samples - calls)
        strata = rng.random(count) < mass
        for positive in strata:
            if positive:
                value = float(problem.sample(rng, True, 1)[0])
                positive_values.append(value)
                weighted = problem.positive_scale * value / mass
            else:
                value = float(problem.sample(rng, False, 1)[0])
                negative_values.append(value)
                weighted = -problem.negative_scale * value / (1 - mass)
            calls += 1
            evidence.observe(weighted, plus_bet, minus_bet)
            trace.append((calls, mass, plus_bet, minus_bet, weighted, evidence.log_positive, evidence.log_negative))
            if evidence.decision:
                break
    truth = int(np.sign(problem.integral))
    wrong = evidence.decision != 0 and (truth == 0 or evidence.decision != truth)
    return {
        "problem": problem.name,
        "method": method,
        "seed": seed,
        "calls": calls,
        "decision": evidence.decision,
        "truth": truth,
        "wrong": bool(wrong),
        "certified": evidence.decision != 0,
        "log_positive": evidence.log_positive,
        "log_negative": evidence.log_negative,
        "trace": trace,
    }


def bounded_empirical_bet(values, lower_bound):
    if lower_bound >= 0:
        raise ValueError("A strictly negative lower bound is required")
    bets = np.linspace(0, 0.99, 32) / -lower_bound
    growth = np.log1p(bets[:, None] * np.asarray(values[-64:])[None, :]).mean(axis=1)
    index = int(growth.argmax())
    return float(bets[index]), float(growth[index])


def run_stratified_trial(problem, method, seed, max_samples=4096, alpha=0.05, batch_size=32):
    if method not in {"paired", "upper_bound"}:
        raise ValueError("Unknown stratified method")
    rng = np.random.default_rng(seed)
    positive = list(problem.sample(rng, True, 2))
    negative = list(problem.sample(rng, False, 2))
    a, b = problem.positive_scale, problem.negative_scale
    differences = list(a * np.asarray(positive) - b * np.asarray(negative))
    evidence = EvidenceState(alpha)
    calls = 4
    trace = []
    while calls < max_samples and evidence.decision == 0:
        if method == "paired":
            if calls + 2 > max_samples:
                break
            plus_bet, _ = bounded_empirical_bet(differences, -b)
            minus_bet, _ = bounded_empirical_bet([-value for value in differences], -a)
            for _ in range(min(batch_size // 2, (max_samples - calls) // 2)):
                value = a * float(problem.sample(rng, True, 1)[0]) - b * float(problem.sample(rng, False, 1)[0])
                differences.append(value)
                calls += 2
                evidence.observe(value, plus_bet, minus_bet)
                trace.append((calls, 0.5, plus_bet, minus_bet, value, evidence.log_positive, evidence.log_negative))
                if evidence.decision:
                    break
        else:
            plus_bet, plus_growth = bounded_empirical_bet([a * value - b for value in positive], -b)
            minus_bet, minus_growth = bounded_empirical_bet([b * value - a for value in negative], -a)
            select_positive = plus_growth > minus_growth or (plus_growth == minus_growth and len(positive) <= len(negative))
            for _ in range(min(batch_size, max_samples - calls)):
                value = float(problem.sample(rng, select_positive, 1)[0])
                if select_positive:
                    positive.append(value)
                    weighted = a * value - b
                    evidence.observe(weighted, plus_bet, 0)
                    used_plus, used_minus = plus_bet, 0.0
                else:
                    negative.append(value)
                    weighted = a - b * value
                    evidence.observe(weighted, 0, minus_bet)
                    used_plus, used_minus = 0.0, minus_bet
                calls += 1
                trace.append((calls, float(select_positive), used_plus, used_minus, weighted, evidence.log_positive, evidence.log_negative))
                if evidence.decision:
                    break
    truth = int(np.sign(problem.integral))
    wrong = evidence.decision != 0 and (truth == 0 or evidence.decision != truth)
    return {
        "problem": problem.name, "method": method, "seed": seed, "calls": calls,
        "decision": evidence.decision, "truth": truth, "wrong": bool(wrong),
        "certified": evidence.decision != 0, "log_positive": evidence.log_positive,
        "log_negative": evidence.log_negative, "trace": trace,
    }
