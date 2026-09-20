import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results/continued_iteration_20260921"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise FileExistsError(f"existing artifact differs: {target}")
    else:
        shutil.copy2(source, target)


def main():
    sensor = json.loads((REPORT / "sensor-independent-audit.json").read_text())
    learned = json.loads((REPORT / "learned-independent-audit.json").read_text())
    visual = json.loads((REPORT / "paper-visual-qa.json").read_text())
    render_path = ROOT / "work/tmp/pdfs/continued-final-qa/render-report.json"
    render = json.loads(render_path.read_text())
    pdf_source = ROOT / "work/tmp/pdfs/continued-build/main.pdf"
    pdf_target = ROOT / "output/pdf/ICLR_2027_research_draft_continued_20260921.pdf"
    if sensor["status"] != "passed" or learned["status"] != "passed" or learned["checked_cells"] != 96:
        raise ValueError("independent audit did not pass")
    if visual["status"] != "agent_visual_review_passed" or visual["pdf_sha256"] != digest(pdf_source):
        raise ValueError("visual inspection is not for this PDF")
    if render["pdf_sha256"] != digest(pdf_source) or render["pages"] != 37 or render["unresolved_reference_markers"]:
        raise ValueError("render evidence differs")
    log = (ROOT / "work/tmp/pdfs/continued-build/main.log").read_text()
    if "Overfull" in log or "undefined references" in log or "Citation `" in log:
        raise ValueError("unresolved layout or citation warning")
    tests = (REPORT / "tests-final-cpu.log").read_text()
    if "282 passed, 6 skipped, 6 warnings" not in tests:
        raise ValueError("final test evidence differs")
    preserve_copy(pdf_source, pdf_target)
    preserve_copy(render_path, REPORT / "paper-render-report.json")
    for name in ("main.log", "main.blg", "main.bbl"):
        preserve_copy(ROOT / "work/tmp/pdfs/continued-build" / name, REPORT / ("paper-build-" + name))
    for index in range(1, 5):
        preserve_copy(ROOT / "work/tmp/pdfs/continued-final-qa" / f"sheet-{index}.png", REPORT / f"paper-contact-sheet-{index}.png")
    # 明确区分原始数值审计、开发门槛和人工审稿；文件哈希不替代科学结论。
    paths = sorted(p for p in (ROOT / "manuscript").rglob("*") if p.is_file() and p.suffix in {".tex", ".bib", ".sty", ".bst"})
    paths += [pdf_target, REPORT / "sensor-independent-audit.json", REPORT / "learned-independent-audit.json",
              REPORT / "sensor-development-analysis.json", REPORT / "learned-factor-analysis.json",
              REPORT / "tests-final-cpu.log", REPORT / "paper-visual-qa.json", REPORT / "paper-render-report.json"]
    evidence = dict(status="verified_research_draft", new_development_cells=512, new_confirmation_cells=0,
                    independent_numerical_audits="passed for both complete development cohorts",
                    performance_gates="both failed; no primary candidate promoted", cpu_tests=dict(passed=282, skipped_cuda=6, warnings=6),
                    paper_pages=37, human_peer_review=False, conference_submission=False,
                    artifact_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths})
    output = REPORT / "delivery-verification.json"
    if output.exists():
        if json.loads(output.read_text()) != evidence:
            raise FileExistsError("delivery report differs")
    else:
        output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in evidence.items() if key != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
