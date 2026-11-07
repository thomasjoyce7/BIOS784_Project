import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from construct_labels import construct_labels, CLASSES
from preprocess_metadata import create_splits, predictors
from preprocess_ecg import preprocess_ecg


@pytest.fixture
def mapping():
    return pd.DataFrame({'diagnostic': [1, 1, 0], 'diagnostic_class': ['MI', 'CD', np.nan]},
                        index=['infarct', 'block', 'rhythm'])


def test_multilabel_unknown_likelihood_and_unlabeled(mapping):
    df = pd.DataFrame({'ecg_id': [1, 2, 3],
                       'scp_codes': ["{'infarct': 0.0, 'block': 100.0}", "{'rhythm': 0.0}", '{}']})
    labels = construct_labels(df, mapping)
    assert labels.loc[0, ['MI', 'CD']].tolist() == [1, 1]
    assert labels.loc[1:, CLASSES].to_numpy().sum() == 0
    assert labels.has_diagnostic_label.tolist() == [1, 0, 0]


def test_unknown_scp_code_fails(mapping):
    with pytest.raises(ValueError, match='Unmapped'):
        construct_labels(pd.DataFrame({'ecg_id': [1], 'scp_codes': ["{'UNKNOWN': 100}"]}), mapping)


@pytest.mark.parametrize('folds', [[8, 9], [1, 2]])
def test_patient_in_multiple_folds_fails(folds):
    with pytest.raises(ValueError, match='multiple official folds'):
        create_splits(pd.DataFrame({'ecg_id': [1, 2], 'patient_id': [7, 7], 'strat_fold': folds}))


def test_splits_use_official_folds():
    df = pd.DataFrame({'ecg_id': range(4), 'patient_id': range(4), 'strat_fold': [1, 8, 9, 10]})
    assert create_splits(df).partition.tolist() == ['train', 'train', 'validation', 'test']
    df.loc[0, 'strat_fold'] = 11
    with pytest.raises(ValueError, match='outside'):
        create_splits(df)


def test_predictors_preserve_missing_and_age_and_exclude_targets():
    df = pd.DataFrame({'ecg_id': [1, 2], 'patient_id': [9, 10], 'age': [305., np.nan],
                       'sex': [0, 1], 'height': [np.nan, 170], 'weight': [70, np.nan],
                       'report': ['MI', 'normal'], 'scp_codes': ['{}', '{}'], 'heart_axis': ['x', 'y']})
    pred = predictors(df)
    assert pred.age.iloc[0] == 305 and pd.isna(pred.age.iloc[1])
    assert pred.age_deidentified.iloc[0] and pd.isna(pred.age_deidentified.iloc[1])
    assert pred.height.isna().tolist() == [True, False]
    assert not {'report', 'scp_codes', 'heart_axis'} & set(pred.columns)


@pytest.mark.parametrize('defect', ['nan', 'lead_order', 'frequency', 'shape', 'missing_file'])
def test_waveform_exclusions_preserve_alignment(tmp_path, monkeypatch, defect):
    import preprocess_ecg as module
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    good = np.arange(12000).reshape(1000, 12) / 1000
    def read(path):
        wave = good.copy()
        header = {'fs': 100, 'sig_name': leads.copy(), 'units': ['mV'] * 12}
        if path.endswith('bad'):
            if defect == 'nan':
                wave[0, 0] = np.nan
            elif defect == 'lead_order':
                header['sig_name'][0:2] = ['II', 'I']
            elif defect == 'frequency':
                header['fs'] = 500
            elif defect == 'shape':
                wave = wave[:999]
            else:
                raise FileNotFoundError('missing source')
        return wave, header
    monkeypatch.setattr(module.wfdb, 'rdsamp', read)
    df = pd.DataFrame({'ecg_id': [10, 20, 30], 'filename_lr': ['a', 'bad', 'c']})
    valid, excluded = preprocess_ecg(df, tmp_path, tmp_path,
                                    {'samples': 1000, 'sampling_frequency': 100, 'leads': leads},
                                    tmp_path / 'exclusions.csv')
    assert valid == [10, 30] and excluded[0]['ecg_id'] == 20
    wave = np.load(tmp_path / 'waveforms.npy')
    assert wave.shape == (2, 1000, 12) and wave.dtype == np.float32
    np.testing.assert_allclose(wave[1], good, rtol=1e-7)
    assert np.load(tmp_path / 'recording_ids.npy').tolist() == valid
    assert pd.read_csv(tmp_path / 'exclusions.csv').ecg_id.tolist() == [20]


def test_saved_dataset_validation_detects_corruption(tmp_path):
    import json
    import wfdb
    from validate_dataset import validate
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    raw.mkdir()
    out.mkdir()
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal = (np.arange(12000).reshape(1000, 12) % 1000) / 1000
    wfdb.wrsamp('source', fs=100, units=['mV']*12, sig_name=leads,
                p_signal=signal, fmt=['16']*12, adc_gain=[1000]*12,
                baseline=[0]*12, write_dir=str(raw))
    pd.DataFrame({'ecg_id': [1], 'patient_id': [5], 'age': [np.nan], 'sex': [1],
                  'height': [170.], 'weight': [np.nan], 'strat_fold': [10],
                  'filename_lr': ['source'], 'scp_codes': ["{'NORM': 0}"]}).to_csv(raw / 'ptbxl_database.csv', index=False)
    pd.DataFrame({'diagnostic': [1], 'diagnostic_class': ['NORM']}, index=['NORM']).to_csv(raw / 'scp_statements.csv')
    df = pd.read_csv(raw / 'ptbxl_database.csv')
    predictors(df).to_csv(out / 'predictors.csv', index=False)
    create_splits(df).to_csv(out / 'splits.csv', index=False)
    labels = construct_labels(df, pd.read_csv(raw / 'scp_statements.csv', index_col=0))
    labels.to_csv(out / 'labels.csv', index=False)
    np.save(out / 'waveforms.npy', signal[None].astype('float32'))
    np.save(out / 'recording_ids.npy', np.array([1], dtype='int64'))
    (out / 'dataset.json').write_text(json.dumps({'leads': leads, 'normalization': 'none', 'transformations_fitted': []}))
    assert validate(out, raw)['passed']
    labels.loc[0, 'NORM'] = 0
    labels.to_csv(out / 'labels.csv', index=False)
    with pytest.raises(ValueError, match='Incorrect NORM'):
        validate(out, raw)


def test_official_index_boundary_defect_is_documented_not_modified(tmp_path):
    from preprocess_metadata import validate_record_index
    left, right = 'records100/21000/21837_lr', 'records500/00000/00001_hr'
    source = left + right + '\n'
    (tmp_path / 'RECORDS').write_text(source)
    df = pd.DataFrame({'filename_lr': [left], 'filename_hr': [right]})
    warnings = validate_record_index(tmp_path, df)
    assert len(warnings) == 1 and 'Raw file remains unchanged' in warnings[0]
    assert (tmp_path / 'RECORDS').read_text() == source
    (tmp_path / 'RECORDS').write_text('unexpected/path\n')
    with pytest.raises(ValueError, match='disagree'):
        validate_record_index(tmp_path, df)
