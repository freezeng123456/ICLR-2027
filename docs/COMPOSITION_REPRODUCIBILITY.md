# Reproducing the compositional integrability paper

The paper is `manuscript/main.pdf`; its source is `manuscript/main.tex`. It is a research draft in the official ICLR 2027 layout. The local wrapper labels it as unsubmitted. The four distributed style files are unchanged. No claim of conference submission, acceptance, human review, or learned-model performance is made.

## Experimental provenance

The main protocol was fixed at commit `185e953d57699e678314593985abe20c62d7d129` before its SCNet submission. The supplement was fixed separately at `a66b5e3c37c8a33201d515c128157bb8028554e0` before its submission. It adds 240 tail-control cells and 480 one-step cells; it does not replace any main outcome. Both protocols and source commits are retained in Git. The statistical choices and supplementary timeline are described in `COMPOSITION_PROTOCOL.md` and `INTEGRABILITY_PROTOCOL.md`. These are timestamped local protocols, without a claim of registration in an external registry.

| Run | Source archive SHA-256 | Result archive |
|---|---|---|
| Main | `f162687d06afaa28d9d473f956c202eb26b2db7c7caf2f36637f44d5768164d5` | `composition-main-185e953-results.tar.gz` |
| Supplement | `62850f7b551360ede3f47bff110026904aceb23e2e118f1707882d7c58cb1778` | `composition-supp-a66b5e3-results.tar.gz` |

| Result archive | Bytes | SHA-256 |
|---|---:|---|
| Main | 457091638 | `fe0499d452969af80a65c5f6c0fbb673abe41d5fd3ebcb79ccc07be638a78fcf` |
| Supplement | 122992902 | `2dbe4bb9c3c1229723e2a7be8ab9b9aa8815793ad093d2d2ac63b2882c7628dc` |

On the originating Mac, the two result archives are in `/Users/zenghang/Downloads/`. Their SCNet copies are in `/work/home/zenghang/`, and the unpacked remote roots are `COMPOSITION_MATRIX_185e953` and `COMPOSITION_SUPPLEMENT_a66b5e3`. Every result archive contains its configuration manifest, a SHA-256 file manifest, per-cell results, and the Slurm exit table. Identical large files are represented by internal tar hard links; normal tar extraction restores their original paths. Keep the entire archives together with the repository bundle for offline reproduction.

All formal tasks used one RTX 3080 GPU, four allocated CPU cores, Python 3.10.18, PyTorch 1.12.1, NumPy 1.26.4, double precision, and one OMP/MKL thread. The main job `23785639` used 28 tasks; the supplement `23785828` used 12. Their summed allocated elapsed GPU time was 2.069444 hours, excluding smoke checks and local analysis. Concurrency was capped at four GPUs. The historical interpreter was `/work/home/zenghang/miniconda3/envs/pytorch/bin/python`.

Local independent analysis used Python 3.12 with the exact versions in `requirements-composition-analysis.txt`. The remote sampling environment differs intentionally from the independent analysis environment. CPU and GPU RNG streams and different PyTorch versions need not produce bitwise-identical samples. The recorded remote configuration, source hashes, runtime metadata, seed, and one-step chunk size (65536) identify the executed experiment.

## Recover and independently audit existing results

Start from a clean clone of the delivered Git bundle. Its research branch contains both historical experiment commits and the final analysis. From the repository root, set `COMPOSITION_ARCHIVES` to the directory containing the two archives, then run:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-composition-analysis.txt
mkdir -p work/recovered work/verification
shasum -a 256 "$COMPOSITION_ARCHIVES/composition-main-185e953-results.tar.gz"
shasum -a 256 "$COMPOSITION_ARCHIVES/composition-supp-a66b5e3-results.tar.gz"
tar -xzf "$COMPOSITION_ARCHIVES/composition-main-185e953-results.tar.gz" -C work/recovered
tar -xzf "$COMPOSITION_ARCHIVES/composition-supp-a66b5e3-results.tar.gz" -C work/recovered
.venv/bin/python audit_composition_results.py --root work/recovered/COMPOSITION_MATRIX_185e953 --output work/verification/main
.venv/bin/python audit_composition_results.py --root work/recovered/COMPOSITION_SUPPLEMENT_a66b5e3 --output work/verification/supplement
.venv/bin/python audit_composition_references.py --root work/recovered/COMPOSITION_MATRIX_185e953 --output work/verification/references.json
```

Compare the two archive digests with the table above before extraction. The auditor checks every file hash, source commit and source-file digest, configuration, completion marker, visible GPU, task exit, and benchmark metric. It also computes joint sign-pattern total variation and off-diagonal correlations. One-step diagnostics permit histogram moment bounds and largest-weight consistency checks; the protocol did not retain every one-step particle, so complete raw one-step metric recomputation is not claimed.

The scientific auditor needs Git history because it verifies the runtime source digest against `git show <experiment-commit>:composition_benchmark.py`. A source-only ZIP without these commits is insufficient for this provenance check.

## Regenerate mathematical checks and paper figures

For the complete clean-checkout verification, run `.venv/bin/python verify_composition_checkout.py --output work/clean-verification` before modifying tracked files. This reproduces the two manifests, extracts both historical source trees, reruns two full-path benchmark configurations and three million-particle one-step configurations on CPU, independently checks their metrics, and requires all figure PNGs and aggregate CSVs to match the delivered analysis. It records its source commit and distinguishes CPU reproduction from bitwise GPU equality. The full 1,980-cell GPU matrix is not resubmitted by this command.

```bash
.venv/bin/python composition_benchmark.py --self-check work/verification/estimators.json
.venv/bin/python gaussian_integrability.py --output work/verification/integrability.json
.venv/bin/python gaussian_population_reference.py --output work/verification/gaussian_population
.venv/bin/python operator_audit.py --output work/verification/operator
.venv/bin/python plot_composition_results.py --main results/composition_main_20260909 --supplement results/composition_supplement_20260909 --output work/verification/figures
```

The plot command can instead consume the newly audited `work/verification/main` and `work/verification/supplement` directories. Checked-in CSV files retain all seeds and configurations. Plots summarize pre-specified groups; no unsuccessful method or seed is removed.

Build the paper with Tectonic from the manuscript directory:

```bash
mkdir -p work/paper-build
cd manuscript
tectonic main.tex --outdir ../work/paper-build
```

Tectonic must have its normal TeX support bundle available, downloading it if necessary. Figures are included in both vector PDF and PNG form. The typeset manuscript uses vector figures. Document acceptance includes page rendering and visual inspection, beyond successful compilation.

## Execute the fixed SCNet protocols again

Use separate checkouts of the recorded commits and new output directories. Before submitting, verify the current SCNet account, partition, GPU availability, environment, and compute authorization; the historical shell scripts contain the executed account and interpreter. From the main source directory:

```bash
python run_composition_matrix.py --root "$COMPOSITION_MAIN_OUTPUT" --commit 185e953d57699e678314593985abe20c62d7d129 --manifest-only
sbatch --array=0-27%4 run_composition_scnet.sh "$PWD" "$COMPOSITION_MAIN_OUTPUT" 185e953d57699e678314593985abe20c62d7d129
```

After the main array has completed, from the supplementary source directory:

```bash
python run_composition_supplement.py --root "$COMPOSITION_SUPP_OUTPUT" --commit a66b5e3c37c8a33201d515c128157bb8028554e0 --manifest-only
sbatch --array=0-11%4 run_composition_supplement.sh "$PWD" "$COMPOSITION_SUPP_OUTPUT" a66b5e3c37c8a33201d515c128157bb8028554e0
```

Use the recorded PyTorch environment for both commands. Run the launcher's `smoke` mode before the full arrays when changing environments. Capture the full `sacct` task table with fields `JobID,State,ExitCode,Elapsed,AllocTRES` in each root as pipe-separated `slurm.tsv`, and retain all task logs. `package_composition_results.py` creates the deduplicated archive and recovery manifest after every cell and task marker is complete. An exit marker alone does not replace the independent scientific audit.

## Evidence boundaries

The principal theorem concerns a particular exponential weighted Euler discretization, independent sampling with replacement, finite batch sizes, and scalar or diagonal product tails. General mixture zero-denominator cases are excluded. Exact factor scores isolate subsampling from learned-score error. The experiments support a population-validity result, without establishing universal gains in accuracy, joint coverage, or wall-clock performance. The manuscript explicitly reports the difficult mixture's poor particle approximation and the full discretization's own coarse-grid failures.
