# PTB-XL preprocessing QC

Status: PASS (smoke subset).
Raw: 87,203 files verified; 21,799 recordings; 18,869 patients.
Processed: 64 recordings; 0 exclusions; float32 (64, 1000, 12), mV.

| Partition | Recordings | Patients |
|---|---:|---:|
| train | 41 | 40 |
| validation | 17 | 10 |
| test | 6 | 6 |

| Label | Positive | Fraction |
|---|---:|---:|
| NORM | 44 | 0.6875 |
| MI | 4 | 0.0625 |
| STTC | 6 | 0.0938 |
| CD | 6 | 0.0938 |
| HYP | 2 | 0.0312 |

Multilabel: 3; no diagnostic label: 5.
Patient overlap counts: {'train/validation': 0, 'train/test': 0, 'validation/test': 0}.
All serialized rows and labels checked; all waveform values finite; 32 source round trips (or all if fewer).
No fitted transformations, imputation, filtering, or normalization.

See qc_report.json for missingness, partition label distributions, output sizes/hashes, and warnings.
See exclusions.csv for every excluded recording; header only means no exclusions.

Warnings:
- Official RECORDS has a missing newline between records100/21000/21837_lr and records500/00000/00001_hr; parsed as two paths in memory. Raw file remains unchanged.
- Source ages above 89 are deidentified; source values are preserved.
- All-zero label rows are retained; use has_diagnostic_label for diagnostic-task eligibility.
