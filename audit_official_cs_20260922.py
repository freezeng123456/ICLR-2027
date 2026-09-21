import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from confseq.betting import betting_mart
from confseq.predmix import predmix_empbern_twosided_cs


def audit(root):
    provenance = json.loads((root / "provenance.json").read_text())
    summaries = json.loads((root / "summaries.json").read_text())
    directories = sorted(path.parent for path in root.glob("*/*/seed_*/summary.json"))
    if not (root / "done").is_file() or len(directories) != provenance["expected_cells"] or len(summaries) != len(directories):
        raise ValueError("Incomplete official-baseline run")
    replayed = 0
    total_queries = 0
    collected = []
    for directory in directories:
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        collected.append(summary)
        if not (directory / "done").is_file():
            raise ValueError("Missing cell marker")
        with np.load(directory / "trace.npz", allow_pickle=False) as data:
            trace, observations = data["trace"], data["observations"]
        if trace.shape[1] != 9 or observations.shape[1] != 2 or not np.isfinite(trace).all() or not np.isfinite(observations).all():
            raise ValueError("Invalid CS trace")
        if not np.isin(observations[:, 0], [0, 1]).all() or np.any(observations[:, 1] < 0) or np.any(observations[:, 1] > 1):
            raise ValueError("Invalid stratum observation")
        if len(observations) != summary["calls"] or summary["calls"] > config["budget"]:
            raise ValueError("Query accounting mismatch")
        problem = config["problem"]
        scales = [problem["positive_scale"], problem["negative_scale"]]
        lower, upper = [0.0, 0.0], [1.0, 1.0]
        for row_index, row in enumerate(trace):
            calls, count_positive, count_negative = row[:3].astype(int)
            if calls != count_positive + count_negative or calls > len(observations):
                raise ValueError("CS checkpoint counts disagree")
            prefix = observations[:calls]
            positive = prefix[prefix[:, 0] == 0, 1]
            negative = prefix[prefix[:, 0] == 1, 1]
            if [len(positive), len(negative)] != [count_positive, count_negative]:
                raise ValueError("Unqueried observations entered certificate")
            if row_index:
                previous = trace[row_index - 1]
                selected = int(np.argmax([scales[0] * (previous[5] - previous[3]), scales[1] * (previous[6] - previous[4])]))
                queried = observations[int(previous[0]):calls, 0]
                if not np.all(queried == selected) or len(queried) != min(16, config["budget"] - int(previous[0])):
                    raise ValueError("Allocation used information outside prior checkpoint")
            if np.any(row[3:5] < lower) or np.any(row[5:7] > upper):
                raise ValueError("Running intersection widened")
            if not np.allclose([row[7], row[8]], [scales[0] * row[3] - scales[1] * row[6], scales[0] * row[5] - scales[1] * row[4]], atol=1e-10):
                raise ValueError("Risk interval arithmetic mismatch")
            # 每题固定三个种子重放官方财富或区间，其他单元全部核验查询与停止逻辑。
            if config["seed"] in {2000, 2049, 2099}:
                for index, values in enumerate([positive, negative]):
                    if config["method"] == "betting":
                        for boundary, old, orientation, endpoint in [(row[3 + index], lower[index], 1, 0), (row[5 + index], upper[index], 0, 1)]:
                            if boundary != old and boundary != endpoint:
                                wealth = betting_mart(values, boundary, alpha=0.0125, theta=orientation, trunc_scale=0.99)[-1]
                                if not np.isfinite(wealth) or abs(np.log(wealth / 80)) > 1e-3:
                                    raise ValueError("Root does not match official wealth threshold")
                    else:
                        lo, hi = predmix_empbern_twosided_cs(values, alpha=0.025, running_intersection=True)
                        expected = [max(lower[index], lo[-1]), min(upper[index], hi[-1])]
                        if not np.allclose(expected, [row[3 + index], row[5 + index]], atol=1e-10):
                            raise ValueError("Official empirical Bernstein replay mismatch")
                replayed += 1
            lower, upper = list(row[3:5]), list(row[5:7])
            if row_index < len(trace) - 1 and (row[7] > 0 or row[8] < 0):
                raise ValueError("Baseline ran beyond certified checkpoint")
        final = trace[-1]
        decision = 1 if final[7] > 0 else (-1 if final[8] < 0 else 0)
        pa, pb = problem["positive_beta"]
        na, nb = problem["negative_beta"]
        truth = int(np.sign(scales[0] * pa / (pa + pb) - scales[1] * na / (na + nb)))
        expected = {"decision": decision, "truth": truth, "certified": decision != 0, "wrong": decision != 0 and (truth == 0 or truth != decision)}
        if any(summary[key] != value for key, value in expected.items()):
            raise ValueError("Baseline decision summary is inconsistent")
        total_queries += summary["calls"]
    key = lambda row: (row["problem"], row["method"], row["seed"])
    if sorted(collected, key=key) != sorted(summaries, key=key):
        raise ValueError("Baseline consolidation is inconsistent")
    result = {"passed": True, "audited_cells": len(directories), "total_queries": total_queries, "official_replay_checkpoints": replayed, "official_replay_scope": "seeds 2000, 2049, 2099 for every problem", "full_coverage": "all raw observations, allocation choices, count accounting, interval arithmetic, stopping, summaries"}
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256.json"}
    (root / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root.resolve())
