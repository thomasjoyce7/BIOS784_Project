# Logistic Regression Analysis - Shreya B. 

Main analyses: `logistic_regression_binary.ipynb` and `logistic_regression_multilabel.ipynb`. Both notebooks use the processed PTB-XL ECG dataset to evaluate logistic regression using statistical features extracted from 12-lead waveforms. The binary notebook distinguishes normal (NORM) from myocardial infarction (MI)-related ECGs. The multilabel notebook predicts five diagnostic superclasses (NORM, MI, STTC, CD, HYP), allowing multiple labels per recording. Both examine whether PCA preserves predictive performance; the multilabel analysis also evaluates age and sex. 

## Notebook organisation 

- `logistic_regression_binary.ipynb` has data loading and eligibility checks, 60-feature extraction, logistic regression, PCA comparisons, validation/test evaluation, PCA visualization, and feature importance. 
- `logistic_regression_multilabel.ipynb` has five independent binary classifiers, ECG-only and PCA comparisons, MI feature importance, demographic feature preprocessing, and ECG-only versus ECG + age/sex evaluation. 

## Environment and Longleaf execution

These notebooks run in a Python/Jupyter environment containing numpy, pandas, matplotlib, scikit-learn, and IPython. The processed PTB-XL data must already exist under `data/processed/` in the project repository. Required files are `waveforms.npy`, `recording_ids.npy`, `labels.csv`, `predictors.csv`, and `splits.csv`.

The `DATA_DIR` variable near the beginning of each notebook currently uses a user-specific Longleaf path:

```python
DATA_DIR = Path("/work/users/s/b/sbalaji/BIOS784_Project/data/processed")
```

To reproduce the analyses, update `DATA_DIR` to match the local processed-data directory, then run the notebook cells in order. These analyses use scikit-learn rather than GPU training; no CNN SLURM runner is required. The full waveform array must be accessible, so use a machine or compute session with sufficient memory.

## Scientific choices

- Reuse the processed 100 Hz ECG waveforms (1,000 samples × 12 leads), recording IDs, diagnostic labels, and supplied train/validation/test assignments. Check ECG ID alignment and patient separation across splits.
- Extract five summary statistics from each lead: mean, standard deviation, minimum, maximum, and root mean square (RMS), for 60 predictors per recording. Standardize using training-set statistics only.
- Fit scikit-learn `LogisticRegression` with `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, and `random_state=42`.
- For binary classification, include ECGs labeled NORM but not MI or MI but not NORM. Other diagnostic superclass labels may still be present. Compare the original 60 predictors with PCA retaining at least 95% of training variance (19 PCs in the reported run). Additional PCA dimensions are evaluated on the validation set to investigate the relationship between dimensionality and predictive performance.
- For multilabel classification, exclude ECGs without a diagnostic superclass label. Fit one binary classifier for each of NORM, MI, STTC, CD, and HYP. Compare the original 60 ECG features with 40 principal components, evaluating predictive performance across all five diagnostic superclasses using macro-averaged metrics.
- For the multilabel demographic model, append age and sex to the 60 ECG features (62 predictors). Treat deidentified age placeholders as missing, impute age using the training median, and standardize age using training statistics. Include sex as a binary predictor.
- Use a probability threshold of 0.5 for threshold-based metrics; no class-specific threshold tuning is performed. Examine logistic regression coefficients and validation-set permutation importance to identify ECG features associated with MI prediction. Feature importance reflects predictive associations rather than causal or clinically validated relationships. 
- Reserve the held-out test split for final evaluation rather than model fitting. 

## Held-out est results 

The binary logistic regression model achieved a test AUROC of 0.8306 using the original 60 ECG features, compared with 0.8136 using 19 principal components. PCA substantially reduced dimensionality while preserving most of the model's predictive performance.

For multilabel classification, the ECG-only model achieved a test macro AUROC of 0.7617, macro AUPRC of 0.5516, and macro F1 of 0.4144. Using 40 principal components produced similar results (macro AUROC 0.7611, macro AUPRC 0.5496, macro F1 0.4092). Adding age and sex improved performance, increasing macro AUROC to 0.7924, macro AUPRC to 0.5879, and macro F1 to 0.4770.

These results suggest that PCA can reduce dimensionality with minimal loss of predictive performance, while demographic information provides additional predictive value. Binary AUROC and multilabel macro AUROC evaluate different classification tasks and should not be compared directly.

## Outputs and interpretation

The notebooks embed evaluation tables, plots, and feature-importance results for review in Jupyter or GitHub. The models provide interpretable baselines using waveform summary statistics, rather than the full temporal ECG signal. PCA offers a compact representation with little performance loss; demographic variables add predictive information in the multilabel setting. Performance varies by diagnosis, and the reported metrics do not establish clinical diagnostic validity.