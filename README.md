# BIOS 784 Project: PTB-XL ECG Data

## Downloading and Preprocessing Instructions

Prepare reproducible ECG waveforms, demographic predictors, diagnostic labels,
and patient partitions for future model comparisons. No predictive models are fitted.

Dataset: [PhysioNet PTB-XL **1.0.3**](https://physionet.org/content/ptb-xl/1.0.3/),
DOI 10.13026/kfzx-aw45, CC BY 4.0. Cite the versioned dataset and Wagner et al.,
Scientific Data (2020). Each group member downloads their own copy; data stay out of Git.

Here is an overview of the directory:

```text
configs/                   configurable paths and expected waveform schema
src/                       downloading, preprocessing, and validation
scripts/                   Longleaf SLURM jobs (CPU only)
data/raw/                  complete, unmodified 100 and 500 Hz release
data/processed/            aligned arrays and CSV tables; ignored by Git
reports/preprocessing_qc/  aggregate QC (Quality control) and data dictionaries
tests/                     automated correctness checks
```

### Reproduce on Longleaf

Each group member should log in to Longleaf and clone the repository into their
own work directory. Replace the placeholders below with their directory and the
group's GitHub repository information:

```bash
cd /path/to/your/work/directory
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

If you already cloned the repository, enter that directory instead of cloning it
again. Each member creates their own environment and downloads their own data copy;
the ECG data are not included in the clone.

Use Python 3.11, `curl`, and about 8 GB of free storage. Continue from the cloned
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
sbatch scripts/download.slurm
```

Wait for the download job to complete successfully, then run a smoke test:

```bash
sbatch scripts/preprocess.slurm --limit 64 --output-dir data/smoke --report-dir reports/smoke_qc
```

After smoke QC passes, submit the full run:

```bash
sbatch scripts/preprocess.slurm
```

Submit from the repository root: jobs use `SLURM_SUBMIT_DIR`. Edit SLURM resource
or account directives if your account requires it. Avoid substantial preprocessing
on login nodes. Exact direct commands, for use on a compute node:

```bash
python src/download_ptbxl.py --raw-dir data/raw
python src/create_dataset.py --config configs/preprocessing.yaml
```

YAML paths resolve relative to the repository; absolute paths also work. Override
locations with `--raw-dir`, `--output-dir`, and `--report-dir`. The raw directory
is the parent of `ptb-xl-1.0.3`. Existing nonempty output directories are protected;
choose new output and report directories to rerun. `requirements.txt` pins direct
dependencies; `requirements.lock.txt` records the full tested environment.

### Data conventions

- Waveforms: float32 `(N, 1000, 12)`, 100 Hz, mV, original header lead order
  `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6`. No filtering, resampling,
  normalization, imputation, or fitted transformations. Float32 retains the
  source's 1 microvolt resolution to floating point precision. Keep 500 Hz raw
  recordings for later experiments.
- Predictors: age, sex (0 male, 1 female), height (cm), weight (kg), plus a nullable
  flag for deidentified age above 89. Preserve missing values and source ages near
  300; these are not literal ages. IDs are linkage keys, never model features.
  Diagnostic text, SCP codes, ECG interpretations/quality annotations, and
  annotation workflow fields are excluded. Acquisition fields are also excluded
  from the initial predictors to avoid site/device shortcuts. See the dictionary
  for decisions and missingness for every source field.
- Outcomes: five binary, potentially co-occurring labels: **NORM** normal ECG,
  **MI** myocardial infarction, **STTC** ST/T changes, **CD** conduction disturbance,
  **HYP** hypertrophy. Diagnostic codes are mapped by the official statement file,
  including likelihood zero (unknown likelihood), following the official example.
  All-zero rows are retained with `has_diagnostic_label=0`; apply this eligibility
  flag consistently during later diagnostic modeling instead of treating these
  rows as confirmed controls.
- Splits: official folds **1–8 train, 9 validation, 10 test**. Patient overlap or
  inconsistent folds stop the pipeline. Fit all future learned preprocessing on
  training data only.

### Outputs and checks

Every output has the same ascending recording order. Waveform row `i` corresponds
to CSV row `i` and `recording_ids.npy[i]`.

| File under `data/processed/` | Contents |
|---|---|
| `waveforms.npy` | Memory-mappable physical ECG array |
| `recording_ids.npy` | int64 recording IDs |
| `predictors.csv` | Recording/patient IDs and demographic predictors |
| `labels.csv` | Recording ID, five labels, diagnostic eligibility flag |
| `splits.csv` | Recording/patient IDs, original fold, partition |
| `dataset.json` | Schema, units, config, package versions, source hashes |
| `COMPLETE.json` | Written only after serialized-output validation passes |

### Interpreting and loading waveforms

`waveforms.npy` stores ECG voltage amplitudes as a NumPy array with shape
`(21799, 1000, 12)`: recordings × time points × leads. Each recording contains
10 seconds sampled at 100 Hz (one sample every 0.01 seconds). Each number is a
positive, negative, or zero voltage difference in **millivolts (mV)** for that
lead's electrode configuration. For example, `X[0, 250, 0]` is Lead I in the first
recording at 2.50 seconds; a value of `-0.15` means −0.15 mV. Indices start at zero,
and the lead order is listed above.

The file is about **1.05 GB** because it holds 21,799 × 1,000 × 12 values at
4 bytes per float32 value. Use read-only memory mapping to access recordings
without loading the entire array into RAM. Run this example from the repository root:

```python
import numpy as np

X = np.load("data/processed/waveforms.npy", mmap_mode="r")
recording_ids = np.load("data/processed/recording_ids.npy")

first_ecg = X[0]            # Shape (1000, 12); ID: recording_ids[0]
lead_i = X[0, :, 0]        # Shape (1000,); Lead I over time
time_seconds = np.arange(X.shape[1]) / 100
batch = X[:32]             # First 32 recordings
```

Demographics, labels, and splits are in separate CSV files with the same row
order. Read individual recordings or small batches; selecting a whole partition
with advanced indexing materializes a large copy. The 500 Hz signals remain in
the raw dataset and are not included in this array.

### Quality-control reports and validation

Reports include `qc_report.md`, detailed `qc_report.json`, `exclusions.csv`,
`data_dictionary.csv`, and `outcome_dictionary.csv`. They give sample counts,
missingness, label distributions by partition, patient overlaps, output sizes and
hashes, and warnings. Every excluded ECG has an ID and reason. See
[validation_notes.md](reports/preprocessing_qc/validation_notes.md) for check details
and source definitions.

```bash
cat data/processed/COMPLETE.json
cat reports/preprocessing_qc/qc_report.md
sbatch scripts/test.slurm
# Independently reload and validate on a compute node:
python src/validate_dataset.py --output-dir data/processed --raw-dir data/raw/ptb-xl-1.0.3
```

The downloader resumes partial ZIP transfers, reuses complete cached archives,
checks every official SHA-256 entry, and downloads only damaged/missing files
when repairing an existing release. It saves source, time, manifest digest, and
verified bytes in `data/raw/download_status.json`. The ZIP is removed after
verification. If the archive itself is corrupt, remove only `data/raw/*.zip.part`
and retry. Recheck the raw release on a compute node with:

```bash
python src/download_ptbxl.py --raw-dir data/raw --verify-only
```

Errors return a nonzero status; preprocessing writes `failure.json`. Inspect the
SLURM log and completion marker. Ordinary failures remove staging outputs; killed
jobs may leave `data/.processed-*` staging directories for manual cleanup.

Known upstream 1.0.3 issue: `RECORDS` omits a newline between the last 100 Hz and
first 500 Hz paths. Both files are present and verified. The pipeline parses this
specific boundary in memory and records a QC warning; raw files are untouched.
