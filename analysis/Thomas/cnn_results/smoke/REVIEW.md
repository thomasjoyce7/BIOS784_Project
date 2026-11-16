# CNN preliminary smoke review

Status: PASS. Full training not submitted. Held-out test outcomes not evaluated.

- GPU probe: job 9176722, COMPLETED, exit 0:0.
- GPU notebook smoke: job 9176998, COMPLETED, exit 0:0, 29 seconds SLURM elapsed; about 2.27 GiB peak host RAM.
- Actual environment: PyTorch 2.8.0 + CUDA 12.8; NVIDIA RTX PRO 6000 Blackwell Server Edition.
- 512 training ECGs (508 patients), 256 validation ECGs (252 patients); two trial settings per model, two completed epochs each.
- 11 executed code cells with visible tables and five figures; no execution errors.

## Checks

- PASS: data_alignment_and_tensor_dimensions
- PASS: patient_disjoint_official_splits
- PASS: training_only_preprocessing
- PASS: known_AUROC_AP_F1_and_thresholds
- PASS: weighted_multilabel_loss
- PASS: both_architectures_forward_backward_and_checkpoint_reload
- PASS: whole_lead_and_single_demographic_permutations
- PASS: interrupted_training_resumes_identically
- PASS: CUDA_forward_backward
- PASS: permutation_importance_end_to_end

## Proposed full experiment

Shared four-block CNN, channels 12→32→64→128→256, global average pooling, 64-unit head, five logits. The combined head receives two additional inputs: training-standardized/imputed age and binary sex.
Both models use the same 21,388 diagnostic-eligible recordings (17,084 train / 2,146 validation / 2,158 test), original folds, and training-only per-lead standardization.
AdamW, batch size 128, four prespecified trials per model, up to 50 epochs and patience 8. Epoch/trial selection: validation macro-AUROC. Thresholds: per-class validation F1. Test metrics and model comparison await full execution.

## Timing and resources

Measured training seconds per batch (small smoke subset): 0.007108.
Linear maximum-budget training estimate: 6.4 minutes. With an assumed 2× overhead allowance: 12.7 minutes.
These are extrapolations, not a measured full-run runtime. The final script reserves one GPU, four CPUs, 16 GB host RAM, and one hour for additional headroom.

After review, from the repository root:

```bash
sbatch analysis/Thomas/run_1D_CNN.sh full
```

Remaining methodological choices are documented in CNN_README.md: deidentified-age median imputation, optional training class weighting, AP convention for AUPRC, and validation-only marginal permutation importance. Smoke results do not support scientific conclusions about demographic benefit.
