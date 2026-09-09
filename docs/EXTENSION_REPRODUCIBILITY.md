# Reproducing the sampling and learned-posterior extension

This extension supplements the original 1,980-cell study described in `COMPOSITION_REPRODUCIBILITY.md`. Its fixed source commit is `4b74724c4225fcfe043fdd749fb740fff4cf33c8`. The source archive `composition-extension-20260910.tar.gz` has SHA-256 `d6116b50be251a7d3d58c8748b5c57482b1b0addaba1fc85dba2824889e7bda4`. The frozen protocol is `EXTENSION_PROTOCOL.md`; `EXTENSION_DISPLAY_PLAN.md` fixes presentation choices before formal metrics were downloaded.

The formal extension consists of 1,080 oracle sampling cells, 1,000 learned-density sampling cells and five complete training runs. The 936-configuration without-replacement batch scan is a separate deterministic calculation. Training uses 40,000 updates and 81,920,000 newly simulated pairs for each seed, with fixed final checkpoints. The architecture and controlled simulator are described in the protocol and manuscript appendix.

## Runtime and operational provenance

The completed-attempt output directory on SCNet is `/work/share/acu722p2q8/zenghang_comp_ext_20260910/run`, protected by its owner-only parent directory. It uses the same frozen runtime code as the original extension attempt. The job identifiers are training `23791942`, oracle `23791943` and learned sampling `23791944`, with 5, 40 and 40 array tasks respectively. Each task requests one RTX 3080 and four CPU cores. The historical runtime uses Python 3.10.18, PyTorch 1.12.1 and NumPy 1.26.4. Sampling uses float64; neural network training and parameter prediction use float32, with predicted mixture parameters converted to float64 for sampling.

An earlier attempt in `/work/home/zenghang/composition_extension_20260910` exhausted the user's 50 GB home allocation during file writes. Its jobs were training `23791464`, oracle `23791465` and dependent learned sampling `23791466`. The team already had a separate unused 61 GB shared allocation, so the unchanged protocol was restarted in the private shared directory. Earlier partial files remain in their original directory. Their logs, source provenance and job identifiers are copied into `failed_attempt/` in the complete result archive, and their scheduler records are retained as `failed_attempt_slurm.tsv`. Those partial cells are excluded from formal scientific results. Resource accounting reports the successful and failed attempts separately, and excludes smoke jobs unless stated.

The final archive is `composition_extension_20260910_complete.tar.gz`, exactly 2,797,871,498 bytes, with SHA-256 `037ee3bb2d7eef092e214193af7b41560aae49781ee27d6e7731cc0188010cc7`. It contains all formal raw samples, normalized weights, reference distributions, step diagnostics, configurations, completion records, initial/final neural checkpoints, training histories, scheduler tables and the file-level `recovery_manifest.json`. Exact duplicate files use internal tar hard links, preserving every original path when extracted. All 22,155 file hashes and 85 formal task exits pass the recovery audit. Formal extension tasks used 5.570556 allocated GPU hours; the earlier storage-failure attempt used 1.037778 allocated GPU hours. Both figures exclude smoke jobs.

## Independent audit from raw data

Use a clean clone of the delivered Git bundle, retaining the historical source commit. Install the versions in `requirements-composition-analysis.txt`. Set `COMPOSITION_EXTENSION_ARCHIVE` to the downloaded archive path and verify its SHA-256 against the delivery manifest before extraction.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-composition-analysis.txt
shasum -a 256 "$COMPOSITION_EXTENSION_ARCHIVE"
mkdir -p work/extension-recovered work/extension-analysis
tar -xzf "$COMPOSITION_EXTENSION_ARCHIVE" -C work/extension-recovered
.venv/bin/python audit_extension_archive.py --root work/extension-recovered/run --output work/extension-analysis
.venv/bin/python audit_extension_training.py --root work/extension-recovered/run --output work/extension-analysis
.venv/bin/python audit_extension_results.py --root work/extension-recovered/run --output work/extension-analysis --kind oracle
.venv/bin/python audit_extension_results.py --root work/extension-recovered/run --output work/extension-analysis --kind learned
.venv/bin/python scan_without_replacement.py --output work/extension-analysis
.venv/bin/python plot_extension_results.py --analysis work/extension-analysis --figures work/extension-figures --tables work/extension-tables
```

The archive auditor recomputes every recovered file hash, checks all 85 successful task exits and resource allocations, and compares recorded source hashes with `git show` at the frozen source commit. The scientific auditor reconstructs Gaussian/mixture references using independent SciPy densities on 131,073 points over `[-16,16]`, checks the original 65,537-point references over `[-12,12]`, recomputes particle metrics, and independently applies the Gaussian variance recursion. Identical reference files are checked by hash across methods. Learned-parameter arrays are compared across paired cells and against forward passes of the saved network.

The training audit checks the complete update histories, finite and changed parameters, initial/final checkpoint hashes, and fresh independent validation on 100,000 simulator pairs generated using NumPy seed 944317. It saves those validation pairs and independently integrates conditional KL on the first 1,024 contexts using 16,385 points over `[-16,16]`. These new pairs differ from the original GPU validation stream; their excess negative log likelihood is an independent diagnostic, not a claim of bitwise reconstruction of the original validation data.

## Clean-checkout representative reproduction

Run the following command before modifying a clean clone:

```bash
.venv/bin/python verify_extension_checkout.py --raw-root work/extension-recovered/run --output work/extension-checkout-verification
```

This reproduces both complete manifests, extracts the frozen source, reruns the estimator and Bayes-identity checks, performs a fresh 20-update training smoke check, and executes four representative formal sampling configurations on CPU. Two use the recovered full-training checkpoint and two use oracle factors; they cover unbiased and tail-controlled sampling without replacement. Independent routines verify the rerun metrics. The verifier also regenerates all 936 deterministic certificate configurations, four figure PNGs and the complete grouped tables. It does not repeat the five full GPU trainings or all 2,080 GPU sampling cells, and does not require CPU/GPU random samples to be bitwise identical.

The checked-in extension results and audits are in `results/composition_extension_20260910/`. Passing `--tables` regenerates `learned_tables.tex` directly from all 1,000 audited learned cells. Each of its 40 rows requires exactly the full five-by-five training/dataset seed combination and a common certificate classification. The clean-checkout verifier requires both complete LaTeX tables, including captions, to match the checked-in manuscript byte for byte. Means and sample standard deviations are computed from the individual cells before rounding.

The main learned figure uses all 25 crossed training-seed/dataset-seed pairs at the preselected finer grid, with the coarser grid retained separately. Dispersion across those pairs is not a standard error based on 25 independent training runs. The U sweep holds K fixed and therefore changes the entire time grid; it cannot identify initialization error alone. Timing includes without-replacement subset selection, but neural parameter prediction occurs once per context and is recorded separately.

## Re-executing the full GPU protocol

Use a separate checkout of the frozen commit and a new output directory. Verify the current account, partition, interpreter, resource limits, available storage and compute authorization before submitting. The launcher first executes CPU checks and submits a GPU smoke job. After that job finishes successfully, its matrix stage submits the five trainings and oracle array, followed by the learned array with an `afterok` training dependency:

```bash
python launch_extension.py --root "$COMPOSITION_NEW_RUN" --commit 4b74724c4225fcfe043fdd749fb740fff4cf33c8 --stage smoke
python launch_extension.py --root "$COMPOSITION_NEW_RUN" --commit 4b74724c4225fcfe043fdd749fb740fff4cf33c8 --stage matrix
```

These commands must be run sequentially after checking smoke completion, using the intended GPU runtime. The historical account cap permits at most 40 concurrent GPUs. Once every formal task and training finishes, run the current `recover_extension.py` with the run root and a new archive path, then repeat the independent local audit. The recovery script requires complete markers and successful scheduler exits before packaging.
