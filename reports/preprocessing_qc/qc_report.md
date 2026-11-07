# PTB-XL preprocessing QC

Status: PASS (complete release).
Raw: 87,203 files verified; 21,799 recordings; 18,869 patients.
Processed: 21,799 recordings; 0 exclusions; float32 (21799, 1000, 12), mV.

| Partition | Recordings | Patients |
|---|---:|---:|
| train | 17418 | 15023 |
| validation | 2183 | 1942 |
| test | 2198 | 1904 |

| Label | Positive | Fraction |
|---|---:|---:|
| NORM | 9514 | 0.4364 |
| MI | 5469 | 0.2509 |
| STTC | 5235 | 0.2401 |
| CD | 4898 | 0.2247 |
| HYP | 2649 | 0.1215 |

Multilabel: 5144; no diagnostic label: 411.
Patient overlap counts: {'train/validation': 0, 'train/test': 0, 'validation/test': 0}.
All serialized rows and labels checked; all waveform values finite; 32 source round trips (or all if fewer).
No fitted transformations, imputation, filtering, or normalization.

See qc_report.json for missingness, partition label distributions, output sizes/hashes, and warnings.
See exclusions.csv for every excluded recording; header only means no exclusions.

Warnings:
- Official RECORDS has a missing newline between records100/21000/21837_lr and records500/00000/00001_hr; parsed as two paths in memory. Raw file remains unchanged.
- Source ages above 89 are deidentified; source values are preserved.
- All-zero label rows are retained; use has_diagnostic_label for diagnostic-task eligibility.
