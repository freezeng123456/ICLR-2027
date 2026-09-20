set -euo pipefail
code="$1"
root="$2"
python="$3"
bank="$4"
cd "$code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$root/tmp"
export MPLCONFIGDIR="$root/tmp/mpl"
export XDG_CACHE_HOME="$root/tmp/cache"
export ICLR_TRAINING_ROOT="$bank"
export ICLR_LEARNED_RUN_ROOT="$root/run"
export ICLR_LEARNED_SOURCE="$code"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
test ! -e "$root/audit-independent"
test ! -e "$root/exit-audit.txt"
trap 'status=$?; printf "%s\n" "$status" > "$root/exit-audit.txt"' EXIT
"$python" -B -m pytest -q tests/test_audit_learned_factor_study_20260921.py > "$root/audit-tests.log" 2>&1
"$python" -B audit_learned_factor_study_20260921.py --root "$root/run" --source "$code" --phase development --output "$root/audit-independent" --workers 4 > "$root/audit-development.log" 2>&1
