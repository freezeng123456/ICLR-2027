#!/usr/bin/env bash
set -euo pipefail

research_root=/work/home/zenghang/probability_confirmation_20260922
research_python=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
test "$(cat "$research_root/job.status")" = COMPLETE_PENDING_AUDIT
test ! -e "$research_root/audit.status"
printf 'RUNNING\n' >"$research_root/audit.status"
exec >"$research_root/audit.log" 2>&1
trap 'printf "FAILED\n" >"$research_root/audit.status"' ERR
export GIT_DIR=/work/home/zenghang/ICLR_context_20260909.git
export PYTHONPATH="$research_root/code/work/confseq/src:$research_root/code/work/python-deps"
export MPLCONFIGDIR="$research_root/code/work/matplotlib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$research_root"
sha256sum /work/home/zenghang/audit_matched_confirmation_20260922.py >audit_source.sha256
date -u +'%Y-%m-%dT%H:%M:%SZ'
/usr/bin/time -v "$research_python" /work/home/zenghang/audit_matched_confirmation_20260922.py "$research_root/smoke"
/usr/bin/time -v "$research_python" /work/home/zenghang/audit_matched_confirmation_20260922.py "$research_root/confirmation"
date -u +'%Y-%m-%dT%H:%M:%SZ'
printf 'COMPLETE\n' >"$research_root/audit.status"
