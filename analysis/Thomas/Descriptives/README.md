# Descriptive analysis

Open [descriptive_analysis.ipynb](descriptive_analysis.ipynb) in a compute-backed
Jupyter session and run all cells in order. Use a Python kernel with NumPy, pandas,
and Matplotlib, after completing the shared preprocessing pipeline.

The notebook summarizes demographic predictors and missingness, diagnostic
frequencies, class overlap on the same ECG and across a patient's recordings,
official split separation, and an example ECG's 12 leads and amplitudes. It finds
the repository by its configuration file and memory-maps waveforms, reading one
example recording for the plots. No paths need updating after this folder move.

Existing cell outputs were preserved. Demographic summaries count recordings;
patient-level outcomes use an any-positive-recording rule. Deidentified ages are
flagged and omitted from descriptive age displays without changing the source data.
