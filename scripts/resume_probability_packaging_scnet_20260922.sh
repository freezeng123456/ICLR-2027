#!/usr/bin/env bash
set -euo pipefail

research_root=/work/home/zenghang/probability_confirmation_20260922
research_state=/work/home/zenghang/probability_confirmation_packaging_20260922.status
research_log=/work/home/zenghang/probability_confirmation_packaging_20260922.log
research_archive=/work/home/zenghang/probability_confirmation_20260922.recovery.tar.gz
research_python=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
research_packager=/work/home/zenghang/package_probability_confirmation_20260922_timelimit.py
research_history="$research_root/recovery_history"

test "$(cat "$research_root/job.status")" = COMPLETE_PENDING_AUDIT
test "$(cat "$research_root/audit.status")" = COMPLETE
test "$(cat /work/home/zenghang/probability_confirmation_recovery_20260922.status)" = FAILED
test -s "$research_root/decision_review/analysis.md"
test ! -e "$research_root/recovery_evidence"
test ! -e "$research_history"
test ! -e "$research_state"
test ! -e "$research_log"
test ! -e "$research_archive"
test ! -e "$research_archive.receipt.json"
set -o noclobber
printf 'VALIDATING_AND_PACKAGING\n' >"$research_state"
exec >"$research_log" 2>&1
trap 'printf "FAILED\n" >|"$research_state"' ERR
date -u +'%Y-%m-%dT%H:%M:%SZ'
mkdir "$research_history"
cp /work/home/zenghang/probability_confirmation_recovery_20260922.status "$research_history/"
cp /work/home/zenghang/probability_confirmation_recovery_20260922.log "$research_history/"
cp /work/home/zenghang/probability_confirmation_finalize_launch_20260922.log "$research_history/"
cp /work/home/zenghang/probability_confirmation_slurm_20260922.psv "$research_history/"
cp /work/home/zenghang/finalize_probability_confirmation_scnet_20260922.sh "$research_history/"
cp /work/home/zenghang/package_probability_confirmation_20260922.py "$research_history/"
cp /work/home/zenghang/report_probability_confirmation_20260922.py "$research_history/"
cp /work/home/zenghang/resume_probability_packaging_scnet_20260922.sh "$research_history/"
cp "$research_packager" "$research_history/"
sha256sum "$research_history"/* >"$research_history/SHA256SUMS"
/usr/bin/time -v "$research_python" "$research_packager" --root "$research_root" --archive "$research_archive"
date -u +'%Y-%m-%dT%H:%M:%SZ'
printf 'COMPLETE\n' >|"$research_state"
