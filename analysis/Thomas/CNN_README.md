# 1D CNN analysis: Thomas

Main analysis: `1D_CNN_model.ipynb`. Its default mode is **smoke**. The executed
notebook displays architecture/trainer source, training histories,
metrics, comparison plots, and permutation importance. Smoke figures and metrics
are validation-only plumbing checks, not final predictive results. Review smoke results before deliberately submitting full training.

## Notebook organization

The visible notebook contains data loaders, architecture, training, evaluation,
model comparison, and permutation importance (renumbered 1–6). The former
introduction/configuration/preprocessing/exploration sections and final summary
are removed from view. Required setup, small CSV summaries, runtime records, and
completion bookkeeping run through `cnn_notebook_support.py`. The existing
full-run outputs from job 9177375 and all saved result files are retained verbatim.
No model retraining is needed to view them. The changed notebook layout/support
script requires a fresh smoke review before a future full-mode re-execution; the
original execution records remain historical records of the completed run.

For validation without replacing the main notebook's full-run outputs, set a
separate destination before submitting a smoke job:

```bash
export CNN_OUTPUT_NOTEBOOK=analysis/Thomas/cnn_results/smoke/validation_notebooks/section_cleanup.ipynb
sbatch --time=00:10:00 --mem=16G analysis/Thomas/run_1D_CNN.sh smoke
unset CNN_OUTPUT_NOTEBOOK
```

The execution signature includes the support script as well as notebook code and
`cnn_pipeline.py`; a passed smoke run of that version enables later full reruns.

## Environment and Longleaf execution

The inspected base Python 3.11 environment contains PyTorch 2.8.0 (CUDA 12.8),
NumPy/pandas/scikit-learn, and Jupyter execution libraries. A compute-node probe
successfully performed a CUDA forward/backward pass on an NVIDIA RTX PRO 6000
Blackwell Server Edition. `cnn_requirements.txt` pins the scientific/notebook
versions; runtime `environment.json` records versions, GPU, CUDA, and job ID.

To make an isolated environment rather than using the existing base environment:

```bash
python -m venv .venv-cnn
source .venv-cnn/bin/activate
python -m pip install -r analysis/Thomas/cnn_requirements.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
export CNN_PYTHON="$VIRTUAL_ENV/bin/python"
```

The CUDA wheel command follows [PyTorch's versioned installation instructions](https://pytorch.org/get-started/previous-versions/).
No torchvision or torchaudio is needed. The runner starts its notebook kernel with
the selected Python executable, preventing accidental use of a different kernel.

From the **BIOS784 repository root**, run the small GPU smoke test:

```bash
sbatch --time=00:10:00 --mem=16G analysis/Thomas/run_1D_CNN.sh smoke
```

Check `slurm-cnn-JOBID.out` and `.err`, the SLURM exit status, and
`analysis/Thomas/cnn_results/smoke/smoke_review.json`. A successful smoke run must
show all checks passed and `test_outcomes_evaluated: false`. Execution failures
return a nonzero status and preserve partial notebook outputs and completed-epoch
checkpoints. Hard-killed jobs may leave an incomplete `running` status; rerun to
resume. GPU training requires CUDA and fails rather than falling back to CPU.

**After review**, the exact full-analysis command is:

```bash
sbatch analysis/Thomas/run_1D_CNN.sh full
```

This requests `rtx6k_gpu`, QOS `gpu_rtx6k`, one GPU (`--gres=gpu:1`), four CPUs,
16 GB RAM, and four hours. Those names and the account's QOS were inspected on
Longleaf. Collaborators must confirm their own account has that GPU QOS; partition,
QOS, time, and memory directives can be overridden at submission. The full runner
requires a passed smoke review matching the current notebook code and helper.
Do not run GPU training on a login node or submit smoke and full jobs concurrently
against the same notebook.

## Scientific choices

- Reuse the processed 100 Hz `(N,1000,12)` waveforms, official fold manifest,
  label order NORM/MI/STTC/CD/HYP, and `has_diagnostic_label` eligibility flag.
  Both models use identical eligible ECGs: folds 1–8 train, 9 validation, 10 test.
  Check recording/patient alignment, all official folds, and patient separation.
- Training-only pooled means/SDs standardize each lead on access. No filtering,
  resampling, PCA, or per-recording amplitude normalization is added. Waveforms
  stay memory-mapped instead of being duplicated by partition.
- Only age and sex enter the combined model. Treat age outside 0–89 or flagged
  deidentified as missing; use the training median and standardize after imputation
  with training mean/SD. Sex is 0 male / 1 female, with training-mode imputation for
  missing/invalid values. No ECGs are removed for demographic missingness.
- Same four-block encoder in both models: channels 12→32→64→128→256,
  kernels 7/5/5/3, BatchNorm, ReLU, MaxPool(2), adaptive average pooling. Heads:
  256 (or 258 with demographics) → 64 → 5 logits, ReLU and dropout. AdamW and
  BCEWithLogitsLoss; sigmoid only at inference.
- Four prespecified candidates per model: `(lr, dropout, weight_decay, weighted)` =
  `(1e-3,.3,1e-4,false)`, `(3e-4,.3,1e-4,false)`, `(1e-3,.5,1e-3,false)`,
  `(1e-3,.3,1e-4,true)`. Positive weights use training negatives/positives only.
  Batch size 128, max 50 epochs, patience 8, seed 784. Validation macro-AUROC
  selects epochs, early stopping, and trials. No final retraining on train+validation.
- Validation F1 selects one threshold per class (`p >= t`); ties choose the largest
  threshold. Freeze selected checkpoints and thresholds before test inference.
  Report macro-AUROC, macro-F1, macro-average precision (AP convention for AUPRC),
  F1 at 0.5, and per-class AUROC/AP/F1/precision/recall. Five-class macro means are
  unweighted; undefined class AUROC causes an error rather than a changed denominator.
- Quantify combined minus ECG-only macro-AUROC/F1/AP and each class's AUROC.
  No significance or causal claims are inferred from point estimates.
- Permutation importance uses **validation**, not test: shuffle one entire lead
  across recordings, or age/sex individually, keeping labels fixed. Both models
  use the same lead donor permutations. Five repeats (two for smoke), macro-AUROC
  decrease, mean and repeat SD. Correlation and implausible shuffled combinations
  limit causal interpretation; these are not patient-bootstrap confidence intervals.

## Outputs, resuming, and review

All outputs are under `analysis/Thomas/cnn_results/<run-name>/`: preprocessing and
provenance JSON, checkpoint/status/history files per trial, search comparison CSV,
validation-selected thresholds, prediction caches, aggregate/per-class metrics,
incremental-value tables, permutation summaries, and PNG figures. The notebook
embeds reasonably sized tables/plots for GitHub and does not print patient IDs.
Checkpoints, prediction/linkage NPZ files, environments, and logs are ignored by Git.

Use the same command to resume an interrupted job or reopen completed results.
Latest checkpoints save model/optimizer/RNG/loader state each epoch, and best
checkpoints are separate. Completed trials are reused. A source/data/config change
requires a new run directory to avoid mixing results, for example:

```bash
export CNN_RUN_NAME=full_revision2
sbatch analysis/Thomas/run_1D_CNN.sh full
```

Full execution still requires the **default smoke** review to match current code;
rerun smoke after scientific code changes. If intentionally changing smoke code or
settings after creating its checkpoints, archive the old local `cnn_results/smoke`
directory before rerunning the default smoke command. Do not delete results you
need to preserve. Do not force-push, commit checkpoints, or commit patient-level
prediction artifacts.

The notebook saves measured training-batch timings and an extrapolated maximum
400-epoch budget, with an explicitly assumed 2× overhead allowance. See
`cnn_results/smoke/runtime_estimate.json`; this estimate is not a measured full-run
duration. Early stopping can reduce actual training time.

## Preliminary execution results (November 14, 2026)

GPU probe job 9176722 and notebook smoke job 9176998 both completed with exit 0:0.
The smoke notebook executed all 11 code cells in about 28 seconds (29-second SLURM
job), using about 2.27 GiB peak host memory. All ten scientific/execution checks
passed, including exact interrupted-resume equality with two data-loader workers.
Smoke used 512 training ECGs and 256 validation ECGs, two epochs per trial, two
trial settings per model, and two permutation repeats. No test outcomes were
evaluated. These small-sample scores are not evidence of incremental demographic
benefit.

Measured median training throughput was about 7.1 ms per minibatch after warm-up.
Linear scaling to 134 training batches per epoch and the full 400-epoch maximum
budget suggests about 6.4 minutes of training-only time; the explicitly assumed
2× allowance gives about 12.7 minutes. This can be optimistic because the full
run streams more data, uses different worker behavior, and saves more artifacts.
The four-hour request gives additional headroom; 16 GB host RAM exceeds the smoke
peak plus the approximately 1 GB memory-mapped waveform dataset. Reassess after
the first full execution rather than treating this extrapolation as a guarantee.

Full-scale training uses the explicit `full` submission command above. Test
metrics, final selected configurations, demographic benefit, and scientific
conclusions remain pending successful full execution.
