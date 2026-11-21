# Thomas's analyses

Each analysis has its own folder. All commands below run from the **BIOS784
repository root**, after completing the shared data preprocessing pipeline.

| Folder | Main notebook | Purpose |
|---|---|---|
| [CNN](CNN/CNN_README.md) | [1D_CNN_model.ipynb](CNN/1D_CNN_model.ipynb) | ECG-only vs ECG + age/sex 1D CNNs; GPU |
| [Random_Forest](Random_Forest/README.md) | [Random_Forest_model.ipynb](Random_Forest/Random_Forest_model.ipynb) | 60 ECG summaries vs 60 summaries + age/sex; CPU |
| [Descriptives](Descriptives/README.md) | [descriptive_analysis.ipynb](Descriptives/descriptive_analysis.ipynb) | Demographics, multilabel/patient overlap, example 12-lead ECG |

The existing CNN and descriptive outputs were preserved when these folders were
created. CNN result/checkpoint files moved together into `CNN/cnn_results/`.
Historical job records retain the paths/signatures used when they originally ran;
use the current commands in each analysis README for reruns. Moving the notebook
changes its execution signature, so refresh the CNN smoke check before another
full CNN execution. No CNN models need to be retrained just to view the saved results.

```bash
# CPU random forest checks and small validation-only smoke notebook:
sbatch --partition=interact --time=00:10:00 --cpus-per-task=4 --mem=8G analysis/Thomas/Random_Forest/run_random_forest.sh smoke
# CPU full random forest experiment, after smoke passes:
sbatch analysis/Thomas/Random_Forest/run_random_forest.sh full
# GPU CNN, if a deliberate rerun is needed:
sbatch --time=00:10:00 --mem=16G analysis/Thomas/CNN/run_1D_CNN.sh smoke
sbatch analysis/Thomas/CNN/run_1D_CNN.sh full
```

The RF and CNN share the eligible cohort, folds, label ordering, and validation F1
threshold convention. RF also reports F1 at 0.5 to match the logistic regression
notebook. The CNN uses raw-waveform inputs; the RF and logistic models use the same
60 summary features, so comparisons across all three also reflect different input
representations. Never select hyperparameters or features using test performance.

Commit code, notebooks, aggregate CSV/JSON results, and reasonably sized figures.
Keep datasets, model files, feature/prediction caches, environments, and SLURM logs
out of Git; `.gitignore` covers the generated locations. No automatic commits or
pushes are performed.
