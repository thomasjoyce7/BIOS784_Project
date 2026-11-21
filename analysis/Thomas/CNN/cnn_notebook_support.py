"""Non-displayed setup and completion tasks for the shortened CNN notebook.

The scientific preprocessing/training implementation remains in cnn_pipeline.py.
No work is performed on import; initialize() is called by the first notebook cell.
"""


def initialize():
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from pathlib import Path
    import inspect
    import json
    import sys
    from importlib.metadata import version

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from IPython.display import display, Code, Markdown


    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "configs/preprocessing.yaml").is_file()), None)
    if ROOT is None:
        raise FileNotFoundError("Run from within the BIOS784 repository.")
    ANALYSIS = ROOT / "analysis/Thomas/CNN"
    sys.path.insert(0, str(ANALYSIS))
    from cnn_pipeline import (
        CLASSES, LEADS, MODEL_NAMES, COLORS, SEARCH, ECGCNN, ECGDataset, make_loader,
        load_data, subset_indices, fit_preprocessing, predict, select_thresholds,
        evaluate, search_models, train_trial, smoke_checks, permutation_importance,
        seed_all, signature, write_json, read_json, digest_file, save_figure,
        plot_histories, plot_performance, plot_importance,
    )
    MODE = os.environ.get("CNN_MODE", "smoke")
    if MODE not in {"smoke", "full"}:
        raise ValueError("CNN_MODE must be smoke or full.")
    SMOKE = MODE == "smoke"
    SEED = 784
    BATCH_SIZE = 128
    WORKERS = 0 if SMOKE else 2
    RUN_NAME = os.environ.get("CNN_RUN_NAME", MODE)
    if Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
        raise ValueError("CNN_RUN_NAME must be a single directory name.")
    OUTPUT = ANALYSIS / "cnn_results" / RUN_NAME
    FIGURES = OUTPUT / "figures"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit the GPU SLURM script; do not train on a login node.")
    DEVICE = torch.device("cuda")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(SEED)
    environment = {p: version(p) for p in ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "nbclient", "nbformat"]}
    environment.update(mode=MODE, GPU=torch.cuda.get_device_name(0), CUDA_build=torch.version.cuda,
                       seed=SEED, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    write_json(OUTPUT / "environment.json", environment)

    data = load_data(ROOT)
    indices = subset_indices(data, SEED) if SMOKE else data["indices"]
    parameters = fit_preprocessing(data, indices["train"])
    write_json(OUTPUT / "preprocessing.json", parameters)
    write_json(OUTPUT / "data_provenance.json", data["provenance"])
    # Keep private linkage artifacts local; these files are ignored by Git.
    np.savez_compressed(OUTPUT / "cohort_indices.npz", **indices)

    split_rows, prevalence_rows, demographic_rows = [], [], []
    for partition, rows in indices.items():
        split_rows.append({"partition": partition, "ECGs": len(rows),
                           "patients": data["splits"].iloc[rows].patient_id.nunique()})
        for j, label in enumerate(CLASSES):
            prevalence_rows.append({"partition": partition, "class": label,
                                    "positive": int(data["y"][rows, j].sum()),
                                    "prevalence": float(data["y"][rows, j].mean())})
        raw_demo = data["demographic"][rows]
        demographic_rows.append({"partition": partition,
            "age_missing_or_deidentified": int(np.isnan(raw_demo[:, 0]).sum()),
            "observed_age_mean": float(np.nanmean(raw_demo[:, 0])),
            "observed_age_SD": float(np.nanstd(raw_demo[:, 0], ddof=1)),
            "observed_age_median": float(np.nanmedian(raw_demo[:, 0])),
            "sex_missing_or_invalid": int(np.isnan(raw_demo[:, 1]).sum()),
            "female_percent_observed": float(np.nanmean(raw_demo[:, 1]) * 100)})
    split_summary = pd.DataFrame(split_rows)
    prevalence = pd.DataFrame(prevalence_rows)
    demographic_summary = pd.DataFrame(demographic_rows)
    for name, table in [("cohort_summary", split_summary), ("class_prevalence", prevalence),
                        ("demographic_summary", demographic_summary)]:
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    return locals()


def finalize_run(namespace):
    """Persist runtime/checklist/completion records without a visible summary section."""
    selected = namespace['selected']
    data = namespace['data']
    output = namespace['OUTPUT']
    smoke = namespace['SMOKE']
    np = namespace['np']
    write_json = namespace['write_json']
    measured = [epoch for result in selected.values() for epoch in result['history']]
    warm = [e for e in measured if e['epoch'] > 1] or measured
    seconds_per_batch = float(np.median([e['train_seconds'] / e['batches'] for e in warm]))
    full_batches = int(np.ceil(len(data['indices']['train']) / namespace['BATCH_SIZE']))
    maximum_training_hours = seconds_per_batch * full_batches * 50 * len(namespace['SEARCH']) * 2 / 3600
    runtime = {
        'measured_median_training_seconds_per_batch': seconds_per_batch,
        'full_training_batches_per_epoch': full_batches,
        'maximum_budget_epochs_both_models': 50 * len(namespace['SEARCH']) * 2,
        'extrapolated_training_only_hours': maximum_training_hours,
        'planning_hours_with_2x_allowance': 2 * maximum_training_hours,
        'assumptions': 'Small-subset GPU timing scaled linearly; 2x allowance is assumed, not measured.',
        'estimate_is_extrapolation': True, 'full_run_completed': not smoke,
    }
    write_json(output / 'runtime_estimate.json', runtime)
    if smoke:
        write_json(output / 'smoke_checks.json', {
            'status': 'PASS', 'checks': namespace['checks'],
            'test_outcomes_evaluated': False, 'environment': namespace['environment'],
        })
    write_json(output / 'COMPLETE.json', {
        'status': 'PASS', 'mode': namespace['MODE'],
        'assessment_partition': namespace['partition'],
        'eligible_cohort_hash': data['provenance']['cohort_hash'],
    })
