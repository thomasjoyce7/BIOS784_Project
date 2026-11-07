# Verified execution summary

Executed on Longleaf, November 7, 2026. Download, smoke run, full run, and all 17 tests passed with SLURM exit code 0:0.

Complete PTB-XL 1.0.3: 87,203 manifest files verified (3,173,239,818 bytes), plus the preserved checksum manifest. No missing/corrupt files.
All 21,799 ECGs from 18,869 patients processed; zero exclusions. Waveforms: float32 (21799, 1000, 12), 100 Hz, mV, original 12-lead header order.

| Output | Shape or rows/columns | Bytes |
|---|---|---:|
| `waveforms.npy` | (21799, 1000, 12) | 1,046,352,128 |
| `recording_ids.npy` | (21799,) | 174,520 |
| `predictors.csv` | 21799 rows × 7 columns | 640,094 |
| `labels.csv` | 21799 rows × 7 columns | 381,362 |
| `splits.csv` | 21799 rows × 4 columns | 425,521 |
| `dataset.json` | schema and provenance | 1,845 |

| Partition | Recordings | Patients | Diagnostic-eligible |
|---|---:|---:|---:|
| train | 17,418 | 15,023 | 17,084 |
| validation | 2,183 | 1,942 | 2,146 |
| test | 2,198 | 1,904 | 2,158 |

Official folds 1–8 / 9 / 10 are preserved. All three between-partition patient-overlap counts are zero.

| Label | Positive | Percent | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| NORM | 9,514 | 43.64% | 7,596 | 955 | 963 |
| MI | 5,469 | 25.09% | 4,379 | 540 | 550 |
| STTC | 5,235 | 24.01% | 4,186 | 528 | 521 |
| CD | 4,898 | 22.47% | 3,907 | 495 | 496 |
| HYP | 2,649 | 12.15% | 2,119 | 268 | 262 |

5,144 multilabel records; 411 all-zero label rows retained with diagnostic eligibility false (21,388 eligible).

| Predictor | Missing | Percent |
|---|---:|---:|
| age | 0 | 0.00% |
| sex | 0 | 0.00% |
| height | 14,825 | 68.01% |
| weight | 12,378 | 56.78% |
| age_deidentified | 0 | 0.00% |

293 deidentified ages are flagged; raw age values and missingness are preserved.

Every serialized ID, demographic value/missing value, patient, fold, partition, and label was rechecked. All waveform values are finite; 32 evenly spaced source waveforms were compared after serialization. No transformations were fitted.

Warnings:
- Official RECORDS has a missing newline between records100/21000/21837_lr and records500/00000/00001_hr; parsed as two paths in memory. Raw file remains unchanged.
- Source ages above 89 are deidentified; source values are preserved.
- All-zero label rows are retained; use has_diagnostic_label for diagnostic-task eligibility.

See `qc_report.json` for hashes and full distributions, `data_dictionary.csv` for field definitions and exclusions, and `execution_status.json` for successful job IDs.

Created six Python pipeline modules, three SLURM scripts, YAML configuration, pinned dependency files, 17 automated tests, Git exclusions, and a practical README. No models, commits, or pushes were created.

Reproduce from the repository root (wait for each successful job before the next):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
sbatch scripts/download.slurm
sbatch scripts/preprocess.slurm --limit 64 --output-dir data/smoke --report-dir reports/smoke_qc
sbatch scripts/preprocess.slurm
```

Use fresh output/report locations for repeated runs; existing nonempty output directories are protected. Before modeling, agree on the eligibility mask and handle deidentified ages, missingness, and any learned transformations inside the training pipeline.
