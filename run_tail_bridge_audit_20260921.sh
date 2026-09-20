set -euo pipefail
code="$1"
root="$2"
python="$3"
cd "$code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$root/tmp"
export MPLCONFIGDIR="$root/tmp/mpl"
export XDG_CACHE_HOME="$root/tmp/cache"
mkdir -p "$root/audits" "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
trap 'status=$?; printf "%s\n" "$status" > "$root/exit-audit.txt"' EXIT
"$python" -m pytest -q tests/test_audit_tail_bridge_study_20260921.py > "$root/audit-tests.log" 2>&1
"$python" audit_tail_bridge_study_20260921.py --root "$root/run" --phase development --source "$code" --output "$root/audits/development.json" --workers 4 > "$root/audit-development.log" 2>&1
