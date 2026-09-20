import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from pypdf import PdfReader

from verify_solid_recovery import sha256


def main():
    root = Path(__file__).parent
    output = root / "results/solid_20260920"
    recovery = json.loads((output / "recovery.json").read_text())
    assert recovery["status"] == "passed"
    assert all(sha256(output / name) == value for name, value in recovery["artifact_sha256"].items())
    stats = json.loads((output / "statistics.json").read_text())
    source = (root / "manuscript/solid_results.tex").read_text()
    for method in ["full", "unbiased", "tail"]:
        for name in ["w1_mean", "true_w1_mean"]:
            record = stats["heldout"][method][name]
            for value in [record["mean"], *record["ci95"]]:
                assert f"{value:.5f}" in source
    for row in stats["boundary"]:
        for condition in row["conditions"].values():
            assert f"{condition['w1']['mean']:.5f}" in source
    for row in stats["tail_error"]:
        for condition in row["conditions"].values():
            assert f"{condition['w1']['mean']:.5f}" in source
    with (output / "cells.csv").open() as stream:
        cells = list(csv.DictReader(stream))
    assert len(cells) == 630
    assert sum(row["finite_normalizer"] == "True" for row in cells) == 470
    sensitivity = json.loads((output / "tail_sensitivity.json").read_text())
    assert len(sensitivity) == 66 and sum(row["finite_normalizer"] for row in sensitivity) == 58
    for stem in ["heldout", "boundary", "tail_sensitivity"]:
        assert sha256(output / f"publication_figures/{stem}.pdf") == sha256(root / f"manuscript/figures/solid_{stem}.pdf")
    reader = PdfReader(root / "manuscript/main.pdf")
    pages = [page.extract_text() for page in reader.pages]
    assert len(pages) == 26
    assert "References" in pages[9]
    assert "Frozen replication and tail-misspecification experiments" in pages[22]
    assert not any("??" in page for page in pages)
    log = (root / "work/pdf-check/main.log").read_text()
    assert "Overfull" not in log
    assert not re.search(r"(?:Reference|Citation)[^\n]*undefined", log)
    assert "There were undefined references" not in log
    assert "! LaTeX Error" not in log
    for name in ["local_tests.log", "remote_tests.log"]:
        assert "passed" in (output / name).read_text() and "failed" not in (output / name).read_text()
    files = ["analyze_solid_20260920.py", "verify_solid_publication.py", "verify_solid_recovery.py",
             "manuscript/main.tex", "manuscript/learned_results.tex", "manuscript/solid_results.tex",
             "manuscript/main.pdf", "docs/SOLID_RESULTS_20260920.md"]
    files += [f"manuscript/figures/solid_{stem}.pdf" for stem in ["heldout", "boundary", "tail_sensitivity"]]
    report = dict(status="passed_with_documented_scope_limits", checked_at=datetime.now(timezone.utc).isoformat(),
                  cells=630, finite_sampling_cells=470, infinite_sampling_cells=160,
                  sensitivity_configurations=66, finite_sensitivity_configurations=58,
                  main_text_pages=9, total_pdf_pages=26, appendix_pages=[23, 24, 25, 26],
                  table_values_checked=True, undefined_references=False, overfull_boxes=False,
                  sampling_commit=recovery["sampling_commit"], audit_commit="3df67437832b3f361bf5c0c6de6817bc3e74a48c",
                  visual_review="Main-text pages 8-9 and all four added appendix pages reviewed from rendered PNGs",
                  font_substitution_warnings=[line.strip() for line in log.splitlines() if "Font shape" in line or "Some font shapes" in line],
                  underfull_box_warnings=[line.strip() for line in log.splitlines() if "Underfull" in line],
                  files_sha256={name: sha256(root / name) for name in files},
                  limits=["controlled simulator and explicit Gaussian tails only", "five reused training seeds",
                          "no matched-accuracy speedup demonstrated", "no independent human scientific review",
                          "no external submission or remote git publication performed"])
    (output / "publication_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
