import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exp_conditioning import load_pfn
from train_repro import sha256


CORE_STEMS = ["pfn_cond_w64", "pfn_cond_w64_40k", "pfn_cond", "pfn_cond_40k",
              "pfn_jump_w64", "pfn_jump_w64_40k", "pfn_jump", "pfn_jump_40k"]


def audit(root):
    checkpoints = []
    for path in sorted(root.glob("pfn_*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not state or not all(torch.is_tensor(t) and torch.isfinite(t).all() for t in state.values()):
            raise ValueError(f"非有限或无效的 state_dict: {path}")
        checkpoints.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                            "parameters": sum(t.numel() for t in state.values()),
                            "finite": True})
    cells = []
    for stem in CORE_STEMS:
        checkpoint = root / f"{stem}.pt"
        model, width = load_pfn(checkpoint)
        x = torch.linspace(-1, 1, 24)[None]
        with torch.no_grad():
            mu, lv = model(x, torch.sin(x), 8)
        if mu.shape != (1, 16) or not torch.isfinite(mu).all() or not torch.isfinite(lv).all():
            raise ValueError(f"模型前向检查失败: {checkpoint}")
        prefix = "conditioning" if "cond" in stem else "jump"
        source = root / "results" / f"{prefix}_{stem}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        rows = data["rows"]
        coord = "ell" if prefix == "conditioning" else "rate"
        keys = {(r[coord], r["sigma"], r["n_ctx"], r["design"]) for r in rows}
        if len(rows) != 96 or len(keys) != 96:
            raise ValueError(f"结果缺少唯一的 96 个格子: {source}")
        metrics = np.array([[r[k] for k in ["gap", "gap_se", "dlogvar", "mean_err2"]] for r in rows])
        if not np.isfinite(metrics).all() or (metrics[:, :2] < 0).any():
            raise ValueError(f"主要指标非有限或为负值: {source}")
        groups = {}
        for r in rows:
            groups.setdefault((r[coord], r["sigma"], r["design"]), {})[r["n_ctx"]] = r["gap"]
        rising = sum(g[24] > g[8] for g in groups.values())
        cells.append({"checkpoint": checkpoint.name, "width": width, "result": str(source.relative_to(root)),
                      "result_sha256": sha256(source), "cells": len(rows),
                      "gap_mean": float(metrics[:, 0].mean()), "gap_max": float(metrics[:, 0].max()),
                      "rising_8_to_24": rising, "groups": len(groups),
                      "n_tasks_recorded": data.get("n_tasks"), "quad_check": data["quad_err"]})
    return {"checkpoints": checkpoints, "core_results": cells,
            "limitations": ["checkpoint 与结果文件的历史对应关系没有原始 hash 记录，本次仅建立现存文件清单",
                            "训练 seed、历史环境和训练预算尚未由原始 provenance 独立确认",
                            "验证主要指标和模型前向，不代表重新训练或重新生成全部实验"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"checkpoints": len(result["checkpoints"]),
                      "core_cells": sum(r["cells"] for r in result["core_results"])}))
