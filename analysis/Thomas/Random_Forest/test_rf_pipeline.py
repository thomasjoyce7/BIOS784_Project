"""Contract tests for extraction, leakage controls, metrics, caching, and permutations."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'configs/preprocessing.yaml').exists())
sys.path.insert(0, str(HERE))
import rf_pipeline as rf


def logistic_extractor():
    notebook = json.loads((ROOT / 'analysis/Shreya/logistic_regression_multilabel.ipynb').read_text())
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'code' and source.lstrip().startswith('def extract_ecg_features('):
            tree = ast.parse(source)
            namespace = {'np': np}
            exec(compile(tree, '<logistic feature extractor>', 'exec'), namespace)
            return namespace['extract_ecg_features']
    raise AssertionError('Shared logistic feature extractor not found')


def test_exact_feature_equivalence_to_logistic_notebook():
    x = np.random.default_rng(784).normal(size=(13, 1000, 12)).astype('float32')
    indices = np.array([12, 1, 8, 2, 4, 7])
    expected = logistic_extractor()(x[indices])
    actual = rf.extract_ecg_features(x, indices, batch_size=2)
    np.testing.assert_array_equal(actual, expected)
    assert len(rf.FEATURE_NAMES) == 60 and rf.FEATURE_NAMES[12] == 'Std_I'
    assert rf.FEATURE_NAMES[-1] == 'RMS_V6'


def test_actual_processed_waveform_features_match_logistic_notebook():
    x = np.load(ROOT / 'data/processed/waveforms.npy', mmap_mode='r')
    rows = np.array([0, 100, 5000, len(x)-1])
    np.testing.assert_array_equal(rf.extract_ecg_features(x, rows, 2), logistic_extractor()(x[rows]))


def test_extractor_only_reads_requested_batches():
    class Source:
        calls = []
        def __getitem__(self, rows):
            self.calls.append(len(rows))
            return np.ones((len(rows), 1000, 12), dtype='float32')
    source = Source()
    result = rf.extract_ecg_features(source, np.arange(11), batch_size=3)
    assert source.calls == [3, 3, 3, 2] and result.shape == (11, 60)
    assert np.all(result[:, 12:24] == 0)
    assert np.all(result[:, 48:60] == 1)


def test_invalid_waveform_is_not_silently_dropped():
    x = np.ones((1, 1000, 12), dtype='float32')
    x[0, 50, 4] = np.nan
    with pytest.raises(ValueError, match='Invalid/nonfinite'):
        rf.extract_ecg_features(x, [0])


def demographic_fixture():
    predictors = pd.DataFrame(dict(age=[20, 40, 300, 80, 300], sex=[0, 1, 0, 9, 1],
                                   age_deidentified=[False, False, True, False, True]))
    return dict(predictors=predictors, ids=np.arange(5), indices={'train': np.array([0, 1, 2])})


def test_imputation_is_training_only_and_keeps_all_rows():
    data = demographic_fixture()
    p = rf.fit_demographics(data, [0, 1, 2])
    assert p['age_median'] == 30 and p['sex_mode'] == 0
    changed = demographic_fixture()
    changed['predictors'].loc[[3, 4], 'age'] = -999
    assert rf.fit_demographics(changed, [0, 1, 2]) == p
    indices = {'train': np.array([0, 1, 2]), 'validation': np.array([3, 4])}
    features = {key: np.zeros((len(rows), 60), dtype='float32') for key, rows in indices.items()}
    combined = rf.append_demographics(data, indices, features, p)
    assert combined['validation'].shape == (2, 62)
    np.testing.assert_array_equal(combined['train'][:, -2:], [[20, 0], [40, 1], [30, 0]])
    np.testing.assert_array_equal(combined['validation'][:, -2:], [[80, 0], [30, 1]])
    with pytest.raises(ValueError, match='non-training'):
        rf.fit_demographics(data, [0, 3])


def test_known_metrics_and_validation_thresholds():
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 5))
    p = np.tile(np.array([.1, .4, .35, .8])[:, None], (1, 5))
    thresholds = rf.select_thresholds(y, p)
    np.testing.assert_allclose(thresholds, .35)
    summary, table = rf.evaluate(y, p, thresholds)
    np.testing.assert_allclose([summary['macro_AUROC'], summary['macro_AUPRC_AP'], summary['macro_F1'],
                               summary['macro_F1_at_0_5']], [.75, 5/6, .8, 2/3])
    assert table.class_name.tolist() == rf.CLASSES
    with pytest.raises(ValueError, match='positive and negative'):
        rf.evaluate(np.ones((4, 5)), p, thresholds)


def test_forest_caches_are_reloaded_without_refitting(tmp_path, monkeypatch):
    rng = np.random.default_rng(784)
    x = rng.normal(size=(60, 60)).astype('float32')
    y = rng.integers(0, 2, size=(60, 5), dtype='int8')
    config = dict(rf.SEARCH[0], n_estimators=5)
    forests, before, result = rf.fit_trial(x[:40], y[:40], x[40:], y[40:], config, tmp_path, {}, cores=1)
    def forbidden(*args, **kwargs):
        raise AssertionError('A cached model should not be fitted again')
    monkeypatch.setattr(rf.RandomForestClassifier, 'fit', forbidden)
    restored, after, result2 = rf.fit_trial(x[:40], y[:40], x[40:], y[40:], config, tmp_path, {}, cores=1)
    np.testing.assert_array_equal(before, after)
    assert len(restored) == 5 and result['signature'] == result2['signature']
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'completed'
    with pytest.raises(ValueError, match='stale'):
        rf.fit_trial(x[:40], y[:40], x[40:], y[40:], config, tmp_path, {'changed': True}, cores=1)


def test_grouped_permutation_moves_all_five_lead_features_together(monkeypatch):
    x = np.arange(8*62, dtype='float32').reshape(8, 62)
    y = np.tile([0, 1], 4)[:, None].repeat(5, axis=1)
    observed = []
    def fake_predict(forests, features):
        observed.append(features.copy())
        return np.tile(np.linspace(.1, .9, len(features))[:, None], (1, 5))
    monkeypatch.setattr(rf, 'predict_forests', fake_predict)
    table = rf.grouped_permutation_importance({}, x, y, demographics=True, repeats=2, seed=784)
    assert len(table) == 28
    np.testing.assert_array_equal(observed[0], x)
    groups = [[j+12*k for k in range(5)] for j in range(12)] + [[60], [61]]
    for number, columns in enumerate(groups):
        for repeat in range(2):
            donor = np.random.default_rng(784 + number*1000 + repeat).permutation(8)
            expected = x.copy()
            expected[:, columns] = x[donor][:, columns]
            np.testing.assert_array_equal(observed[1+number*2+repeat], expected)


def test_real_shared_cohort_matches_CNN_and_patient_splits():
    data = rf.load_data(ROOT)
    assert {p: len(rows) for p, rows in data['indices'].items()} == {'train': 17084, 'validation': 2146, 'test': 2158}
    cnn = json.loads((ROOT / 'analysis/Thomas/CNN/cnn_results/full/COMPLETE.json').read_text())
    assert data['provenance']['cohort_hash'] == cnn['eligible_cohort_hash']
    assert data['splits'].groupby('patient_id').partition.nunique().max() == 1
    small = rf.choose_indices(data, smoke=True)
    assert set(small) == {'train', 'validation'}


def test_notebook_preserves_frozen_thresholds_on_roundoff_rerun(tmp_path):
    notebook = json.loads((HERE / 'Random_Forest_model.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'
                  and 'frozen_path = ' in ''.join(c['source']))
    prefix = source[:source.index('display(pd.DataFrame(thresholds')]
    y = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 5))
    p = np.tile(np.array([.1, .4, .35, .8])[:, None], (1, 5))
    namespace = dict(np=np, data={'y': y, 'provenance': {'cohort_hash': 'test'}},
        indices={'validation': np.arange(4)},
        validation_predictions={name: p for name in rf.MODEL_NAMES},
        selected={name: {'signature': name, 'trial': 0} for name in rf.MODEL_NAMES},
        OUTPUT=tmp_path, select_thresholds=rf.select_thresholds, save_json=rf.save_json, read_json=rf.read_json)
    exec(compile(prefix, '<notebook threshold freeze>', 'exec'), namespace)
    before = (tmp_path / 'selection_frozen.json').read_bytes()
    namespace['validation_predictions'] = {name: np.nextafter(p, np.inf) for name in rf.MODEL_NAMES}
    exec(compile(prefix, '<cached notebook threshold freeze>', 'exec'), namespace)
    assert (tmp_path / 'selection_frozen.json').read_bytes() == before
    for name in rf.MODEL_NAMES:
        np.testing.assert_array_equal(namespace['thresholds'][name], [.35]*5)
