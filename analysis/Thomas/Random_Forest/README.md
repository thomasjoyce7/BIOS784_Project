# Random forest ECG analysis

Open [Random_Forest_model.ipynb](Random_Forest_model.ipynb) to view the analysis.
It compares **60 ECG features** with **the same features plus age and sex**, fitting
five independent binary forests in each experiment for NORM, MI, STTC, CD, HYP.
No GPU or PyTorch is required.

## Reproduce on Longleaf

Use Python 3.11 with the pinned scientific/notebook packages in `requirements.txt`.
The inspected base environment already has these versions. To create another:

```bash
python -m venv .venv-rf
source .venv-rf/bin/activate
python -m pip install -r analysis/Thomas/Random_Forest/requirements.txt
export RF_PYTHON="$VIRTUAL_ENV/bin/python"
```

From the **BIOS784 repository root**, first submit the small CPU smoke test:

```bash
sbatch --partition=interact --time=00:10:00 --cpus-per-task=4 --mem=8G analysis/Thomas/Random_Forest/run_random_forest.sh smoke
```

The script runs ten correctness tests and the notebook on 512 training / 256
validation ECGs with two candidates and 20 trees per class. Test outcomes are not
evaluated in smoke mode. Check the SLURM exit status, `slurm-rf-JOBID.out`/`.err`,
and `results/smoke/execution_status.json` for PASS before a full run:

```bash
sbatch analysis/Thomas/Random_Forest/run_random_forest.sh full
```

The full script requests **8 CPUs, 16 GB RAM, 2 hours, no GPU**, on the general
CPU partition. Each forest uses the allocated CPU count; labels and trials run
sequentially. Adjust partition/account directives for your account if necessary.
The runner uses the activated/selected Python for its notebook kernel and returns
nonzero on failure. Do not fit models on a login node.

## Features, cohort, and evaluation

- Reuse processed 100 Hz `(N,1000,12)` mV waveforms. For each lead, calculate
  mean, population SD, minimum, maximum, and RMS over 1,000 samples. The 60 columns
  are statistic-major and use the logistic notebook's lead-name conventions.
  Both synthetic and actual-waveform tests verify **exact numerical equality**
  with its `extract_ecg_features` function. Batch-indexing avoids loading a whole
  partition into RAM. There is no per-record normalization, filtering, PCA, or
  feature selection. Feature scaling is unnecessary for these tree inputs.
- Use the same `has_diagnostic_label=1` cohort as the shared analyses: 17,084 train,
  2,146 validation, 2,158 test ECGs. IDs/order, schema, label order, and all
  patient-disjoint official folds are checked with the shared split procedure.
- Append only age and binary sex (0 male / 1 female) for the combined model.
  Treat absent/invalid/deidentified age as missing and impute the training median;
  missing/invalid sex uses the training mode. Preserve all eligible rows in both
  experiments. The age range/flag policy agrees with the CNN; current privacy-coded
  ages are 300, which the logistic notebook also imputes. No scaling is fitted.
- Four **prespecified**, shared candidates per model, each with 200 trees per class:

  | Trial | Max depth | Min leaf | Max features | Class weight |
  |---|---|---:|---|---|
  | 0 | Unlimited | 1 | sqrt | None |
  | 1 | 20 | 3 | sqrt | balanced_subsample |
  | 2 | Unlimited | 3 | 0.5 | balanced_subsample |
  | 3 | 20 | 5 | sqrt | None |

  Select the highest validation macro-AUROC; ties retain the earlier trial.
  Bootstrap weights use training samples only. Seed 784 is fixed. No refit on
  train+validation is performed. RFs have trees/trials, not neural-network epochs.
- Select one probability threshold per class to maximize validation F1, using
  `p >= threshold` and largest-optimal-threshold tie breaking. Freeze selection
  before test inference. Macro-AUROC, macro-F1, macro-AUPRC (**average precision**),
  fixed-0.5 macro-F1, and per-class AUROC/AP/F1/precision/recall match CNN metric
  conventions. Use the fixed-0.5 F1 column when directly comparing with the current
  logistic notebook. Undefined class AUROC fails rather than changing the denominator.
- Report combined-minus-ECG-only macro and per-class AUROC differences as point
  estimates. These are not statistical significance tests.
- Importance: training impurity importance per class and averaged across classes;
  validation-only grouped permutations of each lead's five summaries, plus age/sex
  individually. Five repeats (two in smoke); corresponding donor permutations match
  between the two models. Repeat SD is not a confidence interval. Correlation and
  implausible cross-lead combinations limit interpretation; no feature selection is
  based on these plots.

## Saved outputs and tests

`results/<run-name>/` contains cohort/prevalence/demographic summaries, feature
dictionary, fitted imputation and provenance, per-trial validation results and
selected settings, frozen thresholds, aggregate/per-class metrics, incremental
metrics, impurity/permutation importance, and PNG figures. The notebook embeds
these results without printing patient identifiers or large arrays.

`models/` stores trusted local joblib estimators/statuses; `.npz` files store local
features, linkage, and predictions. Both are ignored by Git. Never load untrusted
joblib/pickle model files. Code, the executed notebook, small summaries, and figures
can be committed; logs, environments, private caches, and models cannot.

The smoke job runs `test_rf_pipeline.py`: exact feature equivalence, bounded batch
reads, invalid-signal rejection, train-only demographic handling, known AUROC/AP/F1
and thresholds, model reload/stale-cache checks, grouped permutation correctness,
equivalence to the existing CNN cohort, and preservation of already-frozen thresholds on cached reruns. A completed run writes `COMPLETE.json`.
Existing CNN checkpoints and results are not read for RF model development; only
the shared cohort hash is compared in the test.

## Completed full experiment

CPU job **10011831** completed successfully (exit 0:0) in **5 minutes 10 seconds**,
using eight CPU cores and about 2.4 GiB peak host RAM. All eight notebook code
cells executed. The nine correctness tests and the separate smoke notebook also
passed. The final regression suite, including frozen-threshold reuse, passed all
ten tests (job 10014277). No GPU was requested.

| Held-out test metric | ECG only | ECG + age/sex |
|---|---:|---:|
| Macro-AUROC | 0.8091 | 0.8235 |
| Macro-F1 (validation thresholds) | 0.5974 | 0.6023 |
| Macro-AUPRC (AP) | 0.6340 | 0.6457 |
| Macro-F1 at 0.5 | 0.4947 | 0.5516 |

The ECG-only selection used trial 3 (200 trees, max depth 20, minimum leaf 5,
sqrt features, no class weighting). The combined selection used trial 1 (200 trees,
max depth 20, minimum leaf 3, sqrt features, balanced_subsample weights). Both
choices and all thresholds were selected on validation data. These are test point
estimates; statistical significance has not been assessed.

The executed notebook and [full results](results/full/) contain the per-class
metrics, ROC curves, and importance plots. Grouped permutation importance uses
validation records; its plot title was corrected after the completed run without
changing any numerical results or fitted forests. See `results/full/rendering_notes.json`
for that presentation-only correction. A future full re-execution requires the
usual fresh smoke check when notebook code has changed; no rerun is needed to view
the saved full results.
