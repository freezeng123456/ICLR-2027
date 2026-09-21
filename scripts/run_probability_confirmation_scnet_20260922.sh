#!/usr/bin/env bash
set -euo pipefail

research_root="$1"
research_python="$2"
cd "$research_root/code"
trap 'printf "FAILED\n" >"$research_root/job.status"' ERR
printf 'SMOKE\n' >"$research_root/job.status"
"$research_python" -m pytest -q tests/test_decision_evidence_20260922.py tests/test_official_cs_control_20260922.py tests/test_matched_confirmation_20260922.py
"$research_python" run_matched_confirmation_20260922.py --output "$research_root/smoke" --workers 1 --repetitions 1 --seed-start 8000 --budget 100 --problem rare_p0.002_b0.2
printf 'RUNNING\n' >"$research_root/job.status"
"$research_python" run_matched_confirmation_20260922.py --output "$research_root/confirmation" --workers 4 --repetitions 500 --seed-start 9000
printf 'COMPLETE_PENDING_AUDIT\n' >"$research_root/job.status"
