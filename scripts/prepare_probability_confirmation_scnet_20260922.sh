#!/usr/bin/env bash
set -euo pipefail

research_commit="$1"
research_root=/work/home/zenghang/probability_confirmation_20260922
research_bare=/work/home/zenghang/ICLR_context_20260909.git
research_python=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
test ! -e "$research_root"
mkdir -p "$research_root"
exec >"$research_root/prepare.log" 2>&1
trap 'printf "FAILED\n" >"$research_root/prepare.status"' ERR
printf 'PREPARING\n' >"$research_root/prepare.status"
git --git-dir="$research_bare" fetch origin research/iclr-new-direction-20260922
git --git-dir="$research_bare" cat-file -e "$research_commit^{commit}"
git --git-dir="$research_bare" archive --format=tar.gz --output="$research_root/source.tar.gz" "$research_commit"
research_archive_sha=$(sha256sum "$research_root/source.tar.gz" | cut -d ' ' -f 1)
sha256sum "$research_root/source.tar.gz" >"$research_root/source.sha256"
mkdir "$research_root/code"
tar -xzf "$research_root/source.tar.gz" -C "$research_root/code"
cd "$research_root/code"
mkdir -p work/matplotlib work/pip-tmp
git clone https://github.com/gostevehoward/confseq.git work/confseq
test "$(cd work/confseq && git rev-parse HEAD)" = 5ffe733ca2447a2e28c2c91f3b00086173f2ab2c
TMPDIR="$research_root/code/work/pip-tmp" "$research_python" -m pip install --target work/python-deps numpy==1.26.4 scipy==1.13.1 multiprocess==0.70.19 dill==0.4.1 pytest==8.4.2
export PYTHONPATH="$research_root/code/work/confseq/src:$research_root/code/work/python-deps"
export MPLCONFIGDIR="$research_root/code/work/matplotlib"
export CONFSEQ_ROOT="$research_root/code/work/confseq"
export SOURCE_GIT_DIR="$research_bare"
export SOURCE_COMMIT="$research_commit"
export SOURCE_ARCHIVE_SHA256="$research_archive_sha"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
sbatch --parsable --partition=xhhctdnormal --account=acu722p2q8 --ntasks=1 --cpus-per-task=4 --mem=8G --time=01:30:00 --job-name=prob-confirm --output="$research_root/job-%j.log" --export=ALL "$research_root/code/scripts/run_probability_confirmation_scnet_20260922.sh" "$research_root" "$research_python" >"$research_root/job_id"
printf 'SUBMITTED\n' >"$research_root/prepare.status"
