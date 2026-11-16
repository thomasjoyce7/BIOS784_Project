"""Reusable data, training, evaluation, and plotting for the Thomas CNN notebook."""
from pathlib import Path
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_curve)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MODEL_NAMES = ['ECG only', 'ECG + demographics']
COLORS = dict(zip(MODEL_NAMES, ['#277da1', '#d66b27']))
SEARCH = [
    dict(lr=1e-3, dropout=0.3, weight_decay=1e-4, weighted=False),
    dict(lr=3e-4, dropout=0.3, weight_decay=1e-4, weighted=False),
    dict(lr=1e-3, dropout=0.5, weight_decay=1e-3, weighted=False),
    dict(lr=1e-3, dropout=0.3, weight_decay=1e-4, weighted=True),
]


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_data(root):
    root = Path(root)
    folder = root / 'data/processed'
    if not (folder / 'COMPLETE.json').exists():
        raise FileNotFoundError('Run the shared preprocessing pipeline first.')
    schema = read_json(folder / 'dataset.json')
    if (schema['label_order'] != CLASSES or schema['leads'] != LEADS
            or schema['sampling_frequency'] != 100 or schema['units'] != 'mV'
            or schema['normalization'] != 'none' or schema['transformations_fitted']):
        raise ValueError('Unexpected shared waveform/label schema; review before modeling.')
    predictors = pd.read_csv(folder / 'predictors.csv')
    labels = pd.read_csv(folder / 'labels.csv')
    splits = pd.read_csv(folder / 'splits.csv')
    ids = np.load(folder / 'recording_ids.npy', allow_pickle=False)
    wave = np.load(folder / 'waveforms.npy', mmap_mode='r', allow_pickle=False)
    for table in [predictors, labels, splits]:
        if not table.ecg_id.is_unique or not np.array_equal(table.ecg_id, ids):
            raise ValueError('Recording alignment or uniqueness failed.')
    if not np.array_equal(predictors.patient_id, splits.patient_id):
        raise ValueError('Patient alignment failed.')
    if wave.shape != (len(ids), 1000, 12) or wave.dtype != np.float32:
        raise ValueError('Wrong waveform shape/dtype.')
    sys.path.insert(0, str(root / 'src'))
    from preprocess_metadata import create_splits
    expected = create_splits(splits)
    if not expected.equals(splits):
        raise ValueError('Shared official split procedure disagrees with manifest.')
    y = labels[CLASSES].to_numpy(dtype=np.float32)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Nonbinary outcomes.')
    eligible = labels.has_diagnostic_label.to_numpy() == 1
    if not np.array_equal(eligible, y.sum(1) > 0):
        raise ValueError('Shared diagnostic eligibility flag is inconsistent.')
    indices = {p: np.flatnonzero(eligible & splits.partition.eq(p).to_numpy())
               for p in ['train', 'validation', 'test']}
    age = pd.to_numeric(predictors.age, errors='coerce').to_numpy(dtype=float)
    sex = pd.to_numeric(predictors.sex, errors='coerce').to_numpy(dtype=float)
    age_flag = predictors.age_deidentified.fillna(False).to_numpy(dtype=bool)
    age[(age < 0) | (age > 89) | ~np.isfinite(age) | age_flag] = np.nan
    sex[~np.isin(sex, [0, 1])] = np.nan
    demographic = np.column_stack([age, sex])
    cohort_hash = signature({p: ids[i].tolist() for p, i in indices.items()})
    provenance = {'cohort_hash': cohort_hash, 'dataset_json_sha256': digest_file(folder / 'dataset.json'),
                  'table_sha256': {n: digest_file(folder / n) for n in ['predictors.csv', 'labels.csv', 'splits.csv', 'recording_ids.npy']},
                  'waveform_bytes': (folder / 'waveforms.npy').stat().st_size,
                  'helper_sha256': digest_file(__file__)}
    return dict(root=root, wave_path=folder / 'waveforms.npy', wave=wave, y=y,
                predictors=predictors, splits=splits, ids=ids, eligible=eligible,
                demographic=demographic, indices=indices, provenance=provenance)


def subset_indices(data, seed=784, train_n=512, validation_n=256):
    rng = np.random.default_rng(seed)
    result = {}
    for part, n in [('train', train_n), ('validation', validation_n)]:
        source = data['indices'][part]
        result[part] = np.sort(rng.choice(source, min(n, len(source)), replace=False))
        require_binary_classes(data['y'][result[part]])
    return result  # Deliberately contains no test indices.


def require_binary_classes(y):
    if len(y) == 0 or not all(len(np.unique(y[:, j])) == 2 for j in range(5)):
        raise ValueError('Each class needs positives and negatives for five-class macro metrics.')


def fit_preprocessing(data, train_indices, batch_size=128):
    """Fit pooled sample-level lead means/SDs and demographic parameters on train only."""
    if not np.isin(train_indices, data['indices']['train']).all():
        raise ValueError('Preprocessing fit includes a non-training row.')
    count, mean, m2 = 0, np.zeros(12), np.zeros(12)
    for start in range(0, len(train_indices), batch_size):
        x = np.asarray(data['wave'][train_indices[start:start + batch_size]], dtype=np.float64).reshape(-1, 12)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training waveform.')
        n = len(x)
        batch_mean = x.mean(0)
        batch_m2 = np.square(x - batch_mean).sum(0)
        delta = batch_mean - mean
        m2 += batch_m2 + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    std = np.sqrt(m2 / count)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid/constant training lead.')
    demo = data['demographic'][train_indices]
    if not np.isfinite(demo[:, 0]).any() or not np.isfinite(demo[:, 1]).any():
        raise ValueError('Training demographics contain no valid age or sex.')
    age_median = float(np.nanmedian(demo[:, 0]))
    age = np.where(np.isfinite(demo[:, 0]), demo[:, 0], age_median)
    sex_values, sex_counts = np.unique(demo[np.isfinite(demo[:, 1]), 1], return_counts=True)
    age_std = float(age.std())
    if age_std <= 0:
        raise ValueError('Constant training age.')
    y = data['y'][train_indices]
    require_binary_classes(y)
    return dict(lead_mean=mean.tolist(), lead_std=std.tolist(), age_median=age_median,
                age_mean=float(age.mean()), age_std=age_std,
                sex_mode=float(sex_values[np.argmax(sex_counts)]),
                positive_weights=((len(y) - y.sum(0)) / y.sum(0)).tolist(),
                fitted_recordings=len(train_indices), fitted_partition='train',
                training_ids_sha256=signature(data['ids'][train_indices].tolist()),
                age_policy='Source age outside 0–89 or deidentified -> missing; train-median impute',
                sex_policy='0 male / 1 female; invalid -> missing; train-mode impute; not standardized')


def transform_demographics(raw, parameters):
    out = np.array(raw, dtype=np.float32, copy=True)
    out[:, 0] = (np.where(np.isfinite(out[:, 0]), out[:, 0], parameters['age_median'])
                 - parameters['age_mean']) / parameters['age_std']
    out[:, 1] = np.where(np.isfinite(out[:, 1]), out[:, 1], parameters['sex_mode'])
    if not np.isfinite(out).all():
        raise ValueError('Invalid transformed demographics.')
    return out


class ECGDataset(Dataset):
    """Memory-map waveforms lazily in each worker; return (12,1000), (2,), (5,)."""
    def __init__(self, data, indices, parameters, permutation=None):
        self.path = str(data['wave_path'])
        self.indices = np.asarray(indices)
        self.y = data['y']
        self.demographic = transform_demographics(data['demographic'], parameters)
        self.mean = np.asarray(parameters['lead_mean'], dtype=np.float32)
        self.std = np.asarray(parameters['lead_std'], dtype=np.float32)
        self.wave = None
        self.permutation = permutation  # (feature kind, column index, donor local rows)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.wave is None:
            self.wave = np.load(self.path, mmap_mode='r', allow_pickle=False)
        row = self.indices[i]
        x = np.array(self.wave[row], dtype=np.float32, copy=True)
        d = self.demographic[row].copy()
        if self.permutation is not None:
            kind, column, donors = self.permutation
            donor_row = self.indices[donors[i]]
            if kind == 'lead':
                x[:, column] = self.wave[donor_row, :, column]
            elif kind == 'demographic':
                d[column] = self.demographic[donor_row, column]
            else:
                raise ValueError('Unknown permutation kind.')
        x = (x - self.mean) / self.std
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite ECG input.')
        return torch.from_numpy(x.T.copy()), torch.from_numpy(d), torch.from_numpy(self.y[row].copy())

    def __getstate__(self):
        state = self.__dict__.copy()
        state['wave'] = None
        return state


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size=128, shuffle=False, workers=0, seed=784):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      worker_init_fn=seed_worker, generator=generator, pin_memory=True,
                      persistent_workers=False)  # Recreate workers each epoch so generator-state resume is exact.


class ECGCNN(nn.Module):
    """Four-block shared ECG encoder; optional age/sex concatenation before head."""
    def __init__(self, include_demographics=False, dropout=0.3):
        super().__init__()
        self.include_demographics = include_demographics
        blocks = []
        for incoming, outgoing, kernel in [(12, 32, 7), (32, 64, 5), (64, 128, 5), (128, 256, 3)]:
            blocks += [nn.Conv1d(incoming, outgoing, kernel, padding=kernel // 2, bias=False),
                       nn.BatchNorm1d(outgoing), nn.ReLU(), nn.MaxPool1d(2)]
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(258 if include_demographics else 256, 64),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 5))

    def forward(self, ecg, demographics):
        features = self.encoder(ecg)
        if self.include_demographics:
            features = torch.cat([features, demographics], dim=1)
        return self.head(features)  # Logits: no sigmoid or softmax here.


def predict(model, loader, device, return_logits=False):
    model.eval()
    probabilities, targets, all_logits = [], [], []
    with torch.inference_mode():
        for x, d, y in loader:
            logits = model(x.to(device, non_blocking=True), d.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            if return_logits:
                all_logits.append(logits.cpu().numpy())
            targets.append(y.numpy())
    result = (np.concatenate(targets), np.concatenate(probabilities))
    return (*result, np.concatenate(all_logits)) if return_logits else result


def macro_auroc(y, probabilities):
    require_binary_classes(y)
    return float(roc_auc_score(y, probabilities, average='macro'))


def select_thresholds(y_validation, probabilities):
    """Maximize per-class validation F1; tied optima choose the largest threshold."""
    require_binary_classes(y_validation)
    thresholds = []
    for j in range(5):
        precision, recall, candidates = precision_recall_curve(y_validation[:, j], probabilities[:, j])
        denom = precision[:-1] + recall[:-1]
        f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                       out=np.zeros_like(denom), where=denom > 0)
        thresholds.append(float(candidates[np.flatnonzero(f1 == f1.max())[-1]]))
    return np.asarray(thresholds)


def evaluate(y, probabilities, thresholds):
    require_binary_classes(y)
    if probabilities.shape != y.shape or not np.isfinite(probabilities).all():
        raise ValueError('Invalid predicted probabilities.')
    hard = probabilities >= np.asarray(thresholds)
    precision, recall, f1, _ = precision_recall_fscore_support(y, hard, average=None, zero_division=0)
    f1_fixed = precision_recall_fscore_support(y, probabilities >= 0.5, average=None, zero_division=0)[2]
    auc = roc_auc_score(y, probabilities, average=None)
    ap = average_precision_score(y, probabilities, average=None)
    per_class = pd.DataFrame(dict(class_name=CLASSES, AUROC=auc, average_precision=ap,
                                  F1=f1, precision=precision, recall=recall, threshold=thresholds))
    overall = dict(macro_AUROC=float(np.mean(auc)), macro_F1=float(np.mean(f1)),
                   macro_AUPRC_AP=float(np.mean(ap)), macro_F1_at_0_5=float(np.mean(f1_fixed)))
    return overall, per_class


def atomic_torch_save(path, obj):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def trusted_checkpoint(path, device):
    # Only load locally generated checkpoints from this analysis's ignored results directory.
    return torch.load(path, map_location=device, weights_only=False)


def rng_state(loader):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(), loader=loader.generator.get_state())


def restore_rng(state, loader):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    loader.generator.set_state(state['loader'].cpu())


def train_trial(data, indices, parameters, config, include_demographics, folder,
                device, seed=784, batch_size=128, workers=0, max_epochs=50, patience=8, interrupt_after=None):
    """Train/resume one trial. Selection and stopping use validation macro-AUROC only."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(config=config, demographics=include_demographics, seed=seed, batch_size=batch_size,
                    max_epochs=max_epochs, patience=patience, parameters=parameters,
                    train_rows=signature(indices['train'].tolist()), validation_rows=signature(indices['validation'].tolist()),
                    source=data['provenance'])
    fingerprint = signature(identity)
    status_path, latest_path, best_path = folder / 'status.json', folder / 'latest.pt', folder / 'best.pt'
    status = read_json(status_path) if status_path.exists() else {}
    if status and status['fingerprint'] != fingerprint:
        raise ValueError('Run inputs/code/config changed. Choose a new CNN_RUN_NAME; do not reuse stale checkpoints.')
    seed_all(seed)
    model = ECGCNN(include_demographics, config['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    pos_weight = torch.tensor(parameters['positive_weights'], dtype=torch.float32, device=device) if config['weighted'] else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = make_loader(ECGDataset(data, indices['train'], parameters), batch_size, True, workers, seed)
    val_loader = make_loader(ECGDataset(data, indices['validation'], parameters), batch_size, False, workers, seed)
    start_epoch, best_score, best_epoch, stale, history = 0, -float('inf'), 0, 0, []
    if latest_path.exists():
        checkpoint = trusted_checkpoint(latest_path, device)
        if checkpoint['fingerprint'] != fingerprint:
            raise ValueError('Checkpoint fingerprint mismatch.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, best_score, best_epoch, stale, history = [checkpoint[k] for k in
            ['epoch', 'best_score', 'best_epoch', 'stale', 'history']]
        restore_rng(checkpoint['rng'], train_loader)
    done = status.get('status') in ['completed_max_epochs', 'early_stopped']
    completion = status.get('status', 'running')
    if start_epoch >= max_epochs or stale >= patience:
        completion = 'early_stopped' if stale >= patience else 'completed_max_epochs'
        done = True
        write_json(status_path, dict(fingerprint=fingerprint, status=completion, completed_epochs=start_epoch,
                                     best_epoch=best_epoch, best_validation_macro_AUROC=best_score, config=identity))
    if not done:
        try:
            for epoch in range(start_epoch, max_epochs):
                began = time.perf_counter()
                model.train()
                loss_sum, n = 0.0, 0
                for x, d, y in train_loader:
                    x, d, y = [a.to(device, non_blocking=True) for a in (x, d, y)]
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(x, d), y)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss.')
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(y)
                    n += len(y)
                torch.cuda.synchronize(device)
                train_seconds = time.perf_counter() - began
                y_val, p_val, logits_val = predict(model, val_loader, device, return_logits=True)
                # The same loss convention is used for train and validation curves.
                with torch.no_grad():
                    val_loss = criterion(torch.tensor(logits_val, device=device), torch.tensor(y_val, device=device)).item()
                score = macro_auroc(y_val, p_val)
                if score > best_score:
                    best_score, best_epoch, stale = score, epoch + 1, 0
                    atomic_torch_save(best_path, dict(model=model.state_dict(), fingerprint=fingerprint,
                                                     epoch=best_epoch, validation_macro_AUROC=best_score,
                                                     config=identity))
                else:
                    stale += 1
                history.append(dict(epoch=epoch+1, train_loss=loss_sum/n, validation_loss=val_loss,
                                    validation_macro_AUROC=score, train_seconds=train_seconds,
                                    epoch_seconds=time.perf_counter()-began, batches=len(train_loader)))
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                                  best_score=best_score, best_epoch=best_epoch, stale=stale, history=history,
                                  rng=rng_state(train_loader), fingerprint=fingerprint)
                atomic_torch_save(latest_path, checkpoint)
                write_json(folder / 'history.json', history)
                completion = 'early_stopped' if stale >= patience else ('completed_max_epochs' if epoch+1 == max_epochs else 'running')
                write_json(status_path, dict(fingerprint=fingerprint, status=completion,
                                             completed_epochs=epoch+1, best_epoch=best_epoch,
                                             best_validation_macro_AUROC=best_score, config=identity))
                print(f"{'combined' if include_demographics else 'ECG'} trial {folder.name}: epoch {epoch+1}, val AUROC={score:.4f}", flush=True)
                if interrupt_after == epoch + 1:
                    raise KeyboardInterrupt('Intentional interruption for resume smoke test')
                if completion == 'early_stopped':
                    break
        except BaseException as exc:
            write_json(status_path, dict(fingerprint=fingerprint,
                                         status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                                         completed_epochs=len(history), error=type(exc).__name__, config=identity))
            raise
    checkpoint = trusted_checkpoint(best_path, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, dict(status=completion, completed_epochs=len(history), best_epoch=best_epoch,
                       validation_macro_AUROC=best_score, history=history, config=config,
                       checkpoint=str(best_path.relative_to(data['root'])), fingerprint=fingerprint)


def search_models(data, indices, parameters, output, device, smoke=False, workers=0):
    trials = [SEARCH[0], SEARCH[3]] if smoke else SEARCH
    models, selected, rows = {}, {}, []
    for name, demographics in zip(MODEL_NAMES, [False, True]):
        candidates = []
        for trial_number, config in enumerate(trials):
            folder = Path(output) / 'checkpoints' / ('combined' if demographics else 'ecg_only') / f'trial_{trial_number}'
            model, result = train_trial(data, indices, parameters, config, demographics, folder, device,
                                        workers=workers, max_epochs=2 if smoke else 50, patience=2 if smoke else 8)
            candidates.append((result['validation_macro_AUROC'], trial_number, model, result))
            rows.append(dict(model=name, trial=trial_number, **config, **{k: result[k] for k in
                ['status', 'completed_epochs', 'best_epoch', 'validation_macro_AUROC']}))
        # Stable tie handling: earlier trial wins.
        _, _, models[name], selected[name] = max(candidates, key=lambda c: (c[0], -c[1]))
    table = pd.DataFrame(rows)
    table.to_csv(Path(output) / 'hyperparameter_comparison.csv', index=False)
    return models, selected, table


def permutation_importance(model, data, indices, parameters, device, demographics,
                           repeats=5, seed=784, batch_size=128):
    """Validation-only marginal permutations; whole lead trace moves as one unit."""
    if not np.isin(indices, data['indices']['validation']).all():
        raise ValueError('Permutation importance must use validation records.')
    loader = make_loader(ECGDataset(data, indices, parameters), batch_size)
    y, p = predict(model, loader, device)
    baseline = macro_auroc(y, p)
    features = [('lead', j, lead) for j, lead in enumerate(LEADS)]
    if demographics:
        features += [('demographic', 0, 'Age'), ('demographic', 1, 'Sex')]
    rows = []
    for feature_number, (kind, column, label) in enumerate(features):
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + feature_number * 1000 + repeat)
            donors = rng.permutation(len(indices))
            dataset = ECGDataset(data, indices, parameters, permutation=(kind, column, donors))
            y_permuted, p_permuted = predict(model, make_loader(dataset, batch_size), device)
            if not np.array_equal(y, y_permuted):
                raise ValueError('Permutation changed targets/order.')
            score = macro_auroc(y_permuted, p_permuted)
            rows.append(dict(feature=label, kind=kind, repeat=repeat, baseline_macro_AUROC=baseline,
                             permuted_macro_AUROC=score, decrease_macro_AUROC=baseline-score))
    return pd.DataFrame(rows)


def smoke_checks(data, indices, parameters, models, selected, output, device):
    """Exercise scientific contracts on train/validation only; never access test targets."""
    checks = {}
    dataset = ECGDataset(data, indices['train'], parameters)
    x, d, y = dataset[0]
    assert x.shape == (12, 1000) and d.shape == (2,) and y.shape == (5,)
    np.testing.assert_allclose(x.numpy().T,
        (data['wave'][indices['train'][0]] - np.asarray(parameters['lead_mean'])) / np.asarray(parameters['lead_std']),
        rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(y.numpy(), data['y'][indices['train'][0]])
    checks['data_alignment_and_tensor_dimensions'] = True
    checks['patient_disjoint_official_splits'] = True  # load_data already checked the entire manifest.
    # Held-out demographic shifts cannot affect parameters fitted to training rows.
    changed = dict(data)
    changed['demographic'] = data['demographic'].copy()
    changed['demographic'][indices['validation']] = 99999
    again = fit_preprocessing(changed, indices['train'])
    assert parameters == again
    checks['training_only_preprocessing'] = True
    yt = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 5))
    pt = np.tile(np.array([0.1, 0.4, 0.35, 0.8])[:, None], (1, 5))
    thresholds = select_thresholds(yt, pt)
    np.testing.assert_allclose(thresholds, 0.35)
    summary, _ = evaluate(yt, pt, thresholds)
    np.testing.assert_allclose([summary['macro_AUROC'], summary['macro_AUPRC_AP'], summary['macro_F1']],
                               [0.75, 5/6, 0.8])
    checks['known_AUROC_AP_F1_and_thresholds'] = True
    logits = torch.tensor([[0., 1., -1., 2., -2.]], device=device, requires_grad=True)
    target = torch.tensor([[0., 1., 1., 0., 1.]], device=device)
    weights = torch.tensor([1., 2., 3., 4., 5.], device=device)
    actual = nn.BCEWithLogitsLoss(pos_weight=weights)(logits, target)
    manual = -(weights * target * nn.functional.logsigmoid(logits)
               + (1-target) * nn.functional.logsigmoid(-logits)).mean()
    torch.testing.assert_close(actual, manual)
    actual.backward()
    checks['weighted_multilabel_loss'] = True
    sample_loader = make_loader(ECGDataset(data, indices['validation'][:16], parameters), batch_size=8)
    for name, include in zip(MODEL_NAMES, [False, True]):
        seed_all(784)
        fresh = ECGCNN(include).to(device)
        xb, db, yb = next(iter(sample_loader))
        z = fresh(xb.to(device), db.to(device))
        assert z.shape == (len(xb), 5)
        nn.BCEWithLogitsLoss()(z, yb.to(device)).backward()
        assert all(torch.isfinite(p.grad).all() for p in fresh.parameters() if p.grad is not None)
        restored = ECGCNN(include, selected[name]['config']['dropout']).to(device)
        restored.load_state_dict(trusted_checkpoint(data['root'] / selected[name]['checkpoint'], device)['model'])
        restored.eval()
        _, before = predict(models[name], sample_loader, device)
        _, after = predict(restored, sample_loader, device)
        np.testing.assert_allclose(before, after, rtol=0, atol=0)
    checks['both_architectures_forward_backward_and_checkpoint_reload'] = True
    donors = np.arange(len(indices['validation']))[::-1]
    perm = ECGDataset(data, indices['validation'], parameters, ('lead', 0, donors))
    plain = ECGDataset(data, indices['validation'], parameters)
    px, pdemo, py = perm[0]
    donorx, _, _ = plain[int(donors[0])]
    originalx, originald, originaly = plain[0]
    torch.testing.assert_close(px[0], donorx[0])
    torch.testing.assert_close(px[1:], originalx[1:])
    torch.testing.assert_close(pdemo, originald)
    torch.testing.assert_close(py, originaly)
    perm_demo = ECGDataset(data, indices['validation'], parameters, ('demographic', 0, donors))
    dx, dd, dy = perm_demo[0]
    torch.testing.assert_close(dx, originalx)
    torch.testing.assert_close(dd[0], plain[int(donors[0])][1][0])
    torch.testing.assert_close(dd[1], originald[1])
    torch.testing.assert_close(dy, originaly)
    checks['whole_lead_and_single_demographic_permutations'] = True
    # Same seed/inputs: interrupt after epoch 1, resume epoch 2, and compare with uninterrupted training.
    tiny = {p: indices[p][:128] for p in ['train', 'validation']}
    for p in tiny:
        require_binary_classes(data['y'][tiny[p]])
    config = SEARCH[0]
    complete_folder, resume_folder = Path(output) / 'checks/uninterrupted', Path(output) / 'checks/resumed'
    complete, result = train_trial(data, tiny, parameters, config, False, complete_folder, device,
                                   max_epochs=2, patience=2, seed=712, workers=2)
    if not (resume_folder / 'latest.pt').exists():
        try:
            train_trial(data, tiny, parameters, config, False, resume_folder, device,
                        max_epochs=2, patience=2, seed=712, workers=2, interrupt_after=1)
        except KeyboardInterrupt:
            assert read_json(resume_folder / 'status.json')['status'] == 'interrupted'
    resumed, resumed_result = train_trial(data, tiny, parameters, config, False, resume_folder, device,
                                          max_epochs=2, patience=2, seed=712, workers=2)
    assert result['best_epoch'] == resumed_result['best_epoch']
    for key, value in complete.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    # Check the actual last model too, even when the best epoch was the first one.
    complete_last = trusted_checkpoint(complete_folder / 'latest.pt', device)['model']
    resumed_last = trusted_checkpoint(resume_folder / 'latest.pt', device)['model']
    for key in complete_last:
        torch.testing.assert_close(complete_last[key], resumed_last[key], rtol=0, atol=0)
    checks['interrupted_training_resumes_identically'] = True
    checks['CUDA_forward_backward'] = torch.cuda.is_available()
    return checks


def save_figure(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.png', dpi=160, bbox_inches='tight')


def plot_histories(selected, folder):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for name, result in selected.items():
        h = pd.DataFrame(result['history'])
        for ax, metric in zip(axes, ['train_loss', 'validation_loss', 'validation_macro_AUROC']):
            ax.plot(h.epoch, h[metric], label=name, color=COLORS[name])
            ax.set(xlabel='Completed epoch', ylabel=metric.replace('_', ' '))
            ax.axvline(result['best_epoch'], color=COLORS[name], linestyle=':', alpha=.6)
    axes[0].legend()
    fig.suptitle('Selected trial histories; dotted lines mark selected epochs')
    fig.tight_layout()
    save_figure(fig, folder, 'selected_training_histories')
    return fig


def plot_performance(y, probabilities, per_class, folder, partition):
    import matplotlib.pyplot as plt
    figures = []
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for j, (ax, label) in enumerate(zip(axes, CLASSES)):
        for name in MODEL_NAMES:
            fpr, tpr, _ = roc_curve(y[:, j], probabilities[name][:, j])
            ax.plot(fpr, tpr, color=COLORS[name], label=name)
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=.6)
        ax.set(title=label, xlabel='False positive rate', ylabel='True positive rate')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{partition.title()} ROC curves')
    fig.tight_layout()
    save_figure(fig, folder, 'ROC_curves')
    figures.append(fig)
    auc = per_class.pivot(index='class_name', columns='model', values='AUROC').reindex(CLASSES)[MODEL_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    auc.plot.bar(ax=axes[0], color=[COLORS[n] for n in MODEL_NAMES], rot=0)
    axes[0].set(title=f'{partition.title()} per-class AUROC', xlabel='', ylabel='AUROC', ylim=(0, 1))
    (auc[MODEL_NAMES[1]] - auc[MODEL_NAMES[0]]).plot.bar(ax=axes[1], color=COLORS[MODEL_NAMES[1]], rot=0)
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set(title='Increment from age and sex', xlabel='', ylabel='Combined − ECG-only AUROC')
    fig.tight_layout()
    save_figure(fig, folder, 'AUROC_comparison_and_change')
    figures.append(fig)
    return figures


def plot_importance(importance, folder):
    import matplotlib.pyplot as plt
    summary = importance.groupby(['model', 'kind', 'feature']).decrease_macro_AUROC.agg(['mean', 'std']).reset_index()
    summary.to_csv(Path(folder).parent / 'permutation_importance_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, name in zip(axes[:2], MODEL_NAMES):
        s = summary.loc[(summary.model == name) & (summary.kind == 'lead')].set_index('feature').reindex(LEADS)
        ax.barh(s.index, s['mean'], xerr=s['std'].fillna(0), color=COLORS[name], capsize=2)
        ax.axvline(0, color='black', linewidth=.6)
        ax.set(title=name, xlabel='Decrease in validation macro-AUROC')
    s = summary.loc[summary.kind == 'demographic'].set_index('feature').reindex(['Age', 'Sex'])
    axes[2].barh(s.index, s['mean'], xerr=s['std'].fillna(0), color=COLORS[MODEL_NAMES[1]], capsize=2)
    axes[2].axvline(0, color='black', linewidth=.6)
    axes[2].set(title='Combined model: demographics', xlabel='Decrease in validation macro-AUROC')
    fig.suptitle('Marginal permutation importance; error bars = repeat SD, not confidence intervals')
    fig.tight_layout()
    save_figure(fig, folder, 'permutation_importance')
    return fig
