"""CPU random forests using the logistic notebook's 60 ECG summary features."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_curve)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DISPLAY_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
STATISTICS = ['Mean', 'Std', 'Min', 'Max', 'RMS']
FEATURE_NAMES = [f'{stat}_{lead}' for stat in STATISTICS for lead in DISPLAY_LEADS]
MODEL_NAMES = ['ECG only', 'ECG + demographics']
COLORS = dict(zip(MODEL_NAMES, ['#277da1', '#d66b27']))
# Four prespecified candidates, shared by both experiments. No test-driven search.
SEARCH = [
    dict(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features='sqrt', class_weight=None),
    dict(n_estimators=200, max_depth=20, min_samples_leaf=3, max_features='sqrt', class_weight='balanced_subsample'),
    dict(n_estimators=200, max_depth=None, min_samples_leaf=3, max_features=0.5, class_weight='balanced_subsample'),
    dict(n_estimators=200, max_depth=20, min_samples_leaf=5, max_features='sqrt', class_weight=None),
]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_data(root):
    folder = Path(root) / 'data/processed'
    if not (folder / 'COMPLETE.json').is_file():
        raise FileNotFoundError('Complete the shared data pipeline before running this notebook.')
    schema = read_json(folder / 'dataset.json')
    if (schema['label_order'] != CLASSES or schema['leads'] != LEADS
            or schema['sampling_frequency'] != 100 or schema['units'] != 'mV'
            or schema['normalization'] != 'none' or schema['transformations_fitted']):
        raise ValueError('Unexpected processed data schema: raw physical ECGs are required for these features.')
    predictors = pd.read_csv(folder / 'predictors.csv')
    labels = pd.read_csv(folder / 'labels.csv')
    splits = pd.read_csv(folder / 'splits.csv')
    ids = np.load(folder / 'recording_ids.npy', allow_pickle=False)
    wave_path = folder / 'waveforms.npy'
    wave = np.load(wave_path, mmap_mode='r', allow_pickle=False)
    for table in [predictors, labels, splits]:
        if not table.ecg_id.is_unique or not np.array_equal(table.ecg_id, ids):
            raise ValueError('Recording IDs are duplicated or misaligned.')
    if not np.array_equal(predictors.patient_id, splits.patient_id) or splits.patient_id.isna().any():
        raise ValueError('Missing/misaligned patient identifiers.')
    if wave.shape != (len(ids), 1000, 12) or wave.dtype != np.float32:
        raise ValueError('Unexpected waveform shape/dtype.')
    sys.path.insert(0, str(Path(root) / 'src'))
    from preprocess_metadata import create_splits
    if not create_splits(splits).equals(splits):
        raise ValueError('Official fold/partition inconsistency.')
    y = labels[CLASSES].to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Outcomes must be five binary labels.')
    eligible = labels.has_diagnostic_label.eq(1).to_numpy()
    if not np.array_equal(eligible, y.sum(1) > 0):
        raise ValueError('Shared diagnostic eligibility flag is inconsistent.')
    indices = {p: np.flatnonzero(eligible & splits.partition.eq(p).to_numpy())
               for p in ['train', 'validation', 'test']}
    provenance = dict(cohort_hash=signature({p: ids[i].tolist() for p, i in indices.items()}),
        helper_sha256=file_hash(__file__), dataset_sha256=file_hash(folder / 'dataset.json'),
        table_hashes={name: file_hash(folder / name) for name in
                     ['predictors.csv', 'labels.csv', 'splits.csv', 'recording_ids.npy']},
        waveform_size=wave_path.stat().st_size, waveform_mtime_ns=wave_path.stat().st_mtime_ns,
        numpy_version=np.__version__, sklearn_version=sklearn.__version__)
    return dict(root=Path(root), wave=wave, ids=ids, y=y, predictors=predictors,
                splits=splits, eligible=eligible, indices=indices, provenance=provenance)


def choose_indices(data, smoke, seed=784):
    if not smoke:
        return data['indices']
    rng = np.random.default_rng(seed)
    result = {p: np.sort(rng.choice(data['indices'][p], min(n, len(data['indices'][p])), replace=False))
              for p, n in [('train', 512), ('validation', 256)]}
    for rows in result.values():
        check_classes(data['y'][rows])
    return result  # No test subset is exposed to smoke training/evaluation.


def extract_ecg_features(waveforms, indices, batch_size=256):
    """Exact five-statistic, statistic-major representation from Shreya's LR notebook.

    Index within batches, so advanced indexing never copies an entire split.
    """
    indices = np.asarray(indices)
    features = np.empty((len(indices), 60), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        batch = np.asarray(waveforms[rows], dtype=np.float32)
        if batch.shape != (len(rows), 1000, 12) or not np.isfinite(batch).all():
            raise ValueError('Invalid/nonfinite ECG waveform.')
        values = np.concatenate([batch.mean(axis=1), batch.std(axis=1), batch.min(axis=1),
                                 batch.max(axis=1), np.sqrt(np.mean(batch**2, axis=1))], axis=1)
        if not np.isfinite(values).all():
            raise ValueError('Invalid ECG features.')
        features[start:start + len(rows)] = values
    return features


def build_features(data, indices, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for part, rows in indices.items():
        cache = output / f'features_{part}.npz'
        identity = signature(dict(provenance=data['provenance'], ids=data['ids'][rows].tolist(), columns=FEATURE_NAMES))
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['signature'].item()) != identity or not np.array_equal(saved['recording_ids'], data['ids'][rows]):
                    raise ValueError('Feature cache is stale; use a new RF_RUN_NAME.')
                features = saved['features']
        else:
            features = extract_ecg_features(data['wave'], rows)
            np.savez_compressed(cache, signature=identity, recording_ids=data['ids'][rows], features=features)
        if features.shape != (len(rows), 60) or not np.isfinite(features).all():
            raise ValueError('Invalid feature cache.')
        result[part] = features
    return result


def fit_demographics(data, train_rows):
    if not np.isin(train_rows, data['indices']['train']).all():
        raise ValueError('Demographic fit contains a non-training row.')
    raw = clean_demographics(data['predictors'])
    train = raw[train_rows]
    if not np.isfinite(train[:, 0]).any() or not np.isfinite(train[:, 1]).any():
        raise ValueError('No valid training age/sex for imputation.')
    values, counts = np.unique(train[np.isfinite(train[:, 1]), 1], return_counts=True)
    return dict(age_median=float(np.nanmedian(train[:, 0])), sex_mode=float(values[np.argmax(counts)]),
                fitted_partition='train', training_ids_hash=signature(data['ids'][train_rows].tolist()),
                age_policy='Source age <0, >89, missing, or deidentified -> training median',
                sex_policy='0 male / 1 female; invalid/missing -> training mode', scaling='none; tree inputs do not need scaling')


def clean_demographics(predictors):
    age = pd.to_numeric(predictors.age, errors='coerce').to_numpy(dtype=float)
    sex = pd.to_numeric(predictors.sex, errors='coerce').to_numpy(dtype=float)
    flagged = predictors.age_deidentified.fillna(False).to_numpy(dtype=bool)
    age[(age < 0) | (age > 89) | ~np.isfinite(age) | flagged] = np.nan
    sex[~np.isin(sex, [0, 1])] = np.nan
    return np.column_stack([age, sex])


def append_demographics(data, indices, features, params):
    raw = clean_demographics(data['predictors'])
    raw[:, 0] = np.where(np.isfinite(raw[:, 0]), raw[:, 0], params['age_median'])
    raw[:, 1] = np.where(np.isfinite(raw[:, 1]), raw[:, 1], params['sex_mode'])
    result = {p: np.column_stack([features[p], raw[rows]]).astype(np.float32) for p, rows in indices.items()}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite imputed predictors.')
    return result


def check_classes(y):
    if y.ndim != 2 or y.shape[1] != 5 or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid multilabel targets.')
    if not all(len(np.unique(y[:, j])) == 2 for j in range(5)):
        raise ValueError('Each class needs positive and negative records for five-class macro-AUROC.')


def select_thresholds(y, p):
    check_classes(y)
    thresholds = []
    for j in range(5):
        precision, recall, candidates = precision_recall_curve(y[:, j], p[:, j])
        denom = precision[:-1] + recall[:-1]
        scores = np.divide(2 * precision[:-1] * recall[:-1], denom,
                           out=np.zeros_like(denom), where=denom > 0)
        thresholds.append(float(candidates[np.flatnonzero(scores == scores.max())[-1]]))
    return np.asarray(thresholds)


def evaluate(y, probabilities, thresholds):
    check_classes(y)
    if probabilities.shape != y.shape or not np.isfinite(probabilities).all() or not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError('Invalid prediction probabilities.')
    precision, recall, f1, _ = precision_recall_fscore_support(y, probabilities >= thresholds, average=None, zero_division=0)
    f1_fixed = precision_recall_fscore_support(y, probabilities >= 0.5, average=None, zero_division=0)[2]
    auc = roc_auc_score(y, probabilities, average=None)
    ap = average_precision_score(y, probabilities, average=None)
    overall = dict(macro_AUROC=float(np.mean(auc)), macro_F1=float(np.mean(f1)),
                   macro_AUPRC_AP=float(np.mean(ap)), macro_F1_at_0_5=float(np.mean(f1_fixed)))
    table = pd.DataFrame(dict(class_name=CLASSES, AUROC=auc, average_precision=ap, F1=f1,
                              precision=precision, recall=recall, threshold=thresholds))
    return overall, table


def predict_forests(forests, features):
    probabilities = []
    for name in CLASSES:
        forest = forests[name]
        positive = np.flatnonzero(forest.classes_ == 1)
        if len(positive) != 1:
            raise ValueError('A forest has no positive class.')
        probabilities.append(forest.predict_proba(features)[:, positive[0]])
    return np.column_stack(probabilities)


def fit_trial(x_train, y_train, x_val, y_val, config, folder, identity, cores=4, seed=784):
    """Five independent binary forests; parallelize trees, not labels or trials."""
    check_classes(y_train)
    check_classes(y_val)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = signature(dict(identity=identity, config=config, seed=seed, n_features=x_train.shape[1],
        matrices={name: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                  for name, values in [('train_x', x_train), ('train_y', y_train), ('validation_x', x_val), ('validation_y', y_val)]}))
    status_path = folder / 'status.json'
    previous = read_json(status_path) if status_path.exists() else None
    if previous and previous['signature'] != fingerprint:
        raise ValueError('Model cache is stale; choose a new RF_RUN_NAME.')
    forests, class_history = {}, []
    try:
        for j, label in enumerate(CLASSES):
            path = folder / f'{label}.joblib'
            if path.exists():
                saved = joblib.load(path)  # Trusted, locally generated analysis artifacts only.
                if saved['signature'] != fingerprint:
                    raise ValueError('Saved estimator signature mismatch.')
                forest, elapsed = saved['forest'], saved['fit_seconds']
                forest.n_jobs = cores
            else:
                forest = RandomForestClassifier(**config, n_jobs=cores, random_state=seed,
                                                criterion='gini', bootstrap=True)
                start = time.perf_counter()
                forest.fit(x_train, y_train[:, j])
                elapsed = time.perf_counter() - start
                temporary = path.with_suffix('.tmp')
                joblib.dump(dict(signature=fingerprint, forest=forest, fit_seconds=elapsed), temporary, compress=3)
                temporary.replace(path)
            forests[label] = forest
            class_history.append(dict(class_name=label, trees=len(forest.estimators_), fit_seconds=elapsed))
            save_json(status_path, dict(signature=fingerprint, status='training', completed_classes=len(forests),
                                        config=config, class_history=class_history))
        p = predict_forests(forests, x_val)
        auc = float(roc_auc_score(y_val, p, average='macro'))
        save_json(status_path, dict(signature=fingerprint, status='completed', completed_classes=5,
                                    validation_macro_AUROC=auc, config=config, class_history=class_history))
        return forests, p, dict(signature=fingerprint, validation_macro_AUROC=auc,
                                fit_seconds=sum(c['fit_seconds'] for c in class_history), config=config)
    except BaseException as exc:
        save_json(status_path, dict(signature=fingerprint, status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                    completed_classes=len(forests), error=type(exc).__name__, config=config))
        raise


def search_models(data, indices, experiments, params, output, smoke=False, cores=4, seed=784):
    configs = [{**c, 'n_estimators': 20} for c in SEARCH[:2]] if smoke else SEARCH
    models, selected, val_predictions, rows = {}, {}, {}, []
    identity = dict(data=data['provenance'], imputation=params,
                    train_ids_hash=signature(data['ids'][indices['train']].tolist()),
                    validation_ids_hash=signature(data['ids'][indices['validation']].tolist()))
    for name in MODEL_NAMES:
        best = None
        for trial, config in enumerate(configs):
            x = experiments[name]
            folder = Path(output) / 'models' / ('combined' if name == MODEL_NAMES[1] else 'ecg_only') / f'trial_{trial}'
            forests, p, result = fit_trial(x['train'], data['y'][indices['train']], x['validation'],
                                           data['y'][indices['validation']], config, folder, identity, cores, seed)
            rows.append(dict(model=name, trial=trial, **config,
                             validation_macro_AUROC=result['validation_macro_AUROC'], fit_seconds=result['fit_seconds'], status='completed'))
            print(f'{name}, trial {trial}: validation macro-AUROC {result["validation_macro_AUROC"]:.4f}', flush=True)
            if best is None or result['validation_macro_AUROC'] > best:
                best = result['validation_macro_AUROC']
                models[name], selected[name], val_predictions[name] = forests, dict(result, trial=trial), p
    table = pd.DataFrame(rows)
    table.to_csv(Path(output) / 'hyperparameter_comparison.csv', index=False)
    return models, selected, val_predictions, table


def grouped_permutation_importance(forests, features, y, demographics, repeats=5, seed=784):
    """Validation only: jointly shuffle a lead's five summaries, or age/sex separately."""
    baseline = float(roc_auc_score(y, predict_forests(forests, features), average='macro'))
    groups = [(name, [j + 12*k for k in range(5)], 'lead') for j, name in enumerate(DISPLAY_LEADS)]
    if demographics:
        groups += [('Age', [60], 'demographic'), ('Sex', [61], 'demographic')]
    rows = []
    for number, (name, columns, kind) in enumerate(groups):
        for repeat in range(repeats):
            donors = np.random.default_rng(seed + number*1000 + repeat).permutation(len(features))
            x = features.copy()
            x[:, columns] = features[donors][:, columns]
            score = float(roc_auc_score(y, predict_forests(forests, x), average='macro'))
            rows.append(dict(feature=name, kind=kind, repeat=repeat, baseline_macro_AUROC=baseline,
                             permuted_macro_AUROC=score, decrease_macro_AUROC=baseline-score))
    return pd.DataFrame(rows)


def save_figure(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.png', dpi=180, bbox_inches='tight')


def plot_search(table, folder):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name in MODEL_NAMES:
        rows = table.loc[table.model == name]
        axes[0].plot(rows.trial, rows.validation_macro_AUROC, marker='o', label=name, color=COLORS[name])
        axes[1].plot(rows.trial, rows.fit_seconds, marker='o', label=name, color=COLORS[name])
    axes[0].set(title='Validation hyperparameter comparison', xlabel='Prespecified trial', ylabel='Macro-AUROC')
    axes[1].set(title='Measured fitting time (five forests)', xlabel='Prespecified trial', ylabel='Seconds')
    for ax in axes:
        ax.set_xticks(sorted(table.trial.unique()))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, folder, 'validation_search')
    return fig


def plot_performance(y, predictions, by_class, folder, assessment):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for j, ax in enumerate(axes):
        for name in MODEL_NAMES:
            fpr, tpr, _ = roc_curve(y[:, j], predictions[name][:, j])
            ax.plot(fpr, tpr, color=COLORS[name], label=name)
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=.6)
        ax.set(title=CLASSES[j], xlabel='False positive rate', ylabel='True positive rate')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{assessment}: ROC curves')
    fig.tight_layout()
    save_figure(fig, folder, 'ROC_curves')
    auc = by_class.pivot(index='class_name', columns='model', values='AUROC').reindex(CLASSES)[MODEL_NAMES]
    fig2, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    auc.plot.bar(ax=axes[0], color=[COLORS[n] for n in MODEL_NAMES], rot=0)
    axes[0].set(title=f'{assessment}: per-class AUROC', xlabel='', ylabel='AUROC', ylim=(0, 1))
    (auc[MODEL_NAMES[1]] - auc[MODEL_NAMES[0]]).plot.bar(ax=axes[1], color=COLORS[MODEL_NAMES[1]], rot=0)
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set(title='Increment from age and sex', xlabel='', ylabel='Combined − ECG-only AUROC')
    fig2.tight_layout()
    save_figure(fig2, folder, 'AUROC_comparison_and_change')
    return fig, fig2


def impurity_importance(models):
    rows = []
    for name, forests in models.items():
        names = FEATURE_NAMES + (['Age', 'Sex'] if name == MODEL_NAMES[1] else [])
        for label, forest in forests.items():
            rows.extend(dict(model=name, class_name=label, feature=feature, importance=float(value))
                        for feature, value in zip(names, forest.feature_importances_))
    return pd.DataFrame(rows)


def plot_importance(impurity, permutations, folder, assessment):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, ax in zip(MODEL_NAMES, axes):
        values = impurity.loc[impurity.model == name].groupby('feature').importance.mean().nlargest(15).sort_values()
        ax.barh(values.index, values, color=COLORS[name])
        ax.set(title=name, xlabel='Mean impurity importance across five binary forests')
    fig.suptitle('Training impurity importance: top 15; correlated features can share importance')
    fig.tight_layout()
    save_figure(fig, folder, 'feature_impurity_importance')
    summary = permutations.groupby(['model', 'kind', 'feature']).decrease_macro_AUROC.agg(['mean', 'std']).reset_index()
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for name, ax in zip(MODEL_NAMES, axes[:2]):
        values = summary.loc[(summary.model == name) & (summary.kind == 'lead')].set_index('feature').reindex(DISPLAY_LEADS)
        ax.barh(values.index, values['mean'], xerr=values['std'].fillna(0), color=COLORS[name], capsize=2)
        ax.axvline(0, color='black', linewidth=.6)
        ax.set(title=name, xlabel='Decrease in validation macro-AUROC')
    values = summary.loc[summary.kind == 'demographic'].set_index('feature').reindex(['Age', 'Sex'])
    axes[2].barh(values.index, values['mean'], xerr=values['std'].fillna(0), color=COLORS[MODEL_NAMES[1]], capsize=2)
    axes[2].axvline(0, color='black', linewidth=.6)
    axes[2].set(title='Combined: age and sex', xlabel='Decrease in validation macro-AUROC')
    fig2.suptitle(f'{assessment}: grouped permutation importance (repeat SD, not confidence intervals)')
    fig2.tight_layout()
    save_figure(fig2, folder, 'grouped_permutation_importance')
    return fig, fig2, summary
