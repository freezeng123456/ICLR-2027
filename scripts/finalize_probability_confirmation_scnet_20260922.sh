#!/usr/bin/env bash
set -euo pipefail

research_root=/work/home/zenghang/probability_confirmation_20260922
research_state=/work/home/zenghang/probability_confirmation_recovery_20260922.status
research_log=/work/home/zenghang/probability_confirmation_recovery_20260922.log
research_archive=/work/home/zenghang/probability_confirmation_20260922.recovery.tar.gz
research_python=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
test ! -e "$research_state"
test ! -e "$research_log"
test ! -e "$research_archive"
test ! -e "$research_root/decision_review"
test ! -e "$research_root/recovery_evidence"
printf 'WAITING_FOR_AUDIT\n' >"$research_state"
exec >"$research_log" 2>&1
trap 'printf "FAILED\n" >"$research_state"' ERR
date -u +'%Y-%m-%dT%H:%M:%SZ'
sha256sum /work/home/zenghang/report_probability_confirmation_20260922.py /work/home/zenghang/package_probability_confirmation_20260922.py
while true; do
    research_audit_state=$(cat "$research_root/audit.status")
    case "$research_audit_state" in
        COMPLETE) break ;;
        RUNNING) kill -0 36128; sleep 20 ;;
        *) printf 'Unexpected audit state: %s\n' "$research_audit_state"; false ;;
    esac
done
printf 'RENDERING_REPORT\n' >"$research_state"
"$research_python" /work/home/zenghang/report_probability_confirmation_20260922.py "$research_root"
printf 'PACKAGING\n' >"$research_state"
/usr/bin/time -v "$research_python" /work/home/zenghang/package_probability_confirmation_20260922.py --root "$research_root" --archive "$research_archive"
date -u +'%Y-%m-%dT%H:%M:%SZ'
printf 'COMPLETE\n' >"$research_state"
