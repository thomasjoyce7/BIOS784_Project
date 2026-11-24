# Thomas's analyses

Each analysis has its own folder. All commands below run from the **BIOS784
repository root**, after completing the shared data preprocessing pipeline.

| Folder | Main notebook | Purpose |
|---|---|---|
| [CNN](CNN/CNN_README.md) | [1D_CNN_model.ipynb](CNN/1D_CNN_model.ipynb) | ECG-only vs ECG + age/sex 1D CNNs; GPU |
| [Random_Forest](Random_Forest/README.md) | [Random_Forest_model.ipynb](Random_Forest/Random_Forest_model.ipynb) | 60 ECG summaries vs 60 summaries + age/sex; CPU |
| [Descriptives](Descriptives/README.md) | [descriptive_analysis.ipynb](Descriptives/descriptive_analysis.ipynb) | Demographics, multilabel/patient overlap, example 12-lead ECG |