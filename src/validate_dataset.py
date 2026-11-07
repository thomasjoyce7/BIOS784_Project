"""Reload serialized outputs and independently check alignment and source labels."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from construct_labels import CLASSES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(output, root, check_sources=True):
    output, root = Path(output), Path(root)
    info = json.loads((output / 'dataset.json').read_text())
    ids = np.load(output / 'recording_ids.npy', allow_pickle=False)
    wave = np.load(output / 'waveforms.npy', mmap_mode='r', allow_pickle=False)
    tables = {n: pd.read_csv(output / f'{n}.csv') for n in ['predictors', 'labels', 'splits']}
    require(len(ids) == len(set(ids.tolist())), 'Duplicate waveform IDs')
    require(wave.shape == (len(ids), 1000, 12) and wave.dtype == np.float32,
            'Wrong waveform dimensions or dtype')
    for name, table in tables.items():
        require(np.array_equal(table.ecg_id.to_numpy(), ids), f'{name} IDs are misaligned')
    require(np.array_equal(tables['predictors'].patient_id, tables['splits'].patient_id),
            'Patient IDs are misaligned')
    require(set(tables['predictors'].columns) ==
            {'ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight', 'age_deidentified'},
            'Unexpected predictor columns')
    raw = pd.read_csv(root / 'ptbxl_database.csv').set_index('ecg_id').loc[ids]
    mapping = pd.read_csv(root / 'scp_statements.csv', index_col=0)
    for c in CLASSES:
        source_codes = set(mapping.index[(mapping.diagnostic == 1) & (mapping.diagnostic_class == c)])
        expected = [int(bool(set(ast.literal_eval(s)) & source_codes)) for s in raw.scp_codes]
        require(np.array_equal(tables['labels'][c], expected), f'Incorrect {c} labels')
    require(np.array_equal(tables['labels'].has_diagnostic_label,
                           tables['labels'][CLASSES].sum(axis=1).gt(0).astype(int)),
            'Incorrect eligibility flag')
    require(np.array_equal(tables['splits'].strat_fold, raw.strat_fold), 'Official folds changed')
    require(np.array_equal(tables['splits'].patient_id, raw.patient_id), 'Source patients changed')
    expected_split = np.where(raw.strat_fold <= 8, 'train', np.where(raw.strat_fold == 9, 'validation', 'test'))
    require(np.array_equal(tables['splits'].partition, expected_split), 'Wrong partitions')
    for col in ['age', 'sex', 'height', 'weight']:
        np.testing.assert_allclose(tables['predictors'][col], raw[col], rtol=0, atol=0, equal_nan=True)
    expected_flag = raw.age.gt(89).astype('boolean').mask(raw.age.isna()).reset_index(drop=True)
    observed_flag = tables['predictors'].age_deidentified.astype('boolean')
    require(expected_flag.equals(observed_flag), 'Incorrect age flag')
    require(tables['splits'].groupby('patient_id').partition.nunique().max() == 1,
            'Patient overlap detected')
    for start in range(0, len(ids), 128):
        require(np.isfinite(wave[start:start+128]).all(), 'Invalid serialized waveform values')
    # Check source round trips throughout the array, including both ends.
    source_checks = sorted(set(np.linspace(0, len(ids)-1, min(32, len(ids)), dtype=int)))
    if check_sources:
        for i in source_checks:
            original, header = wfdb.rdsamp(str(root / raw.iloc[i].filename_lr))
            require(header['sig_name'] == info['leads'] and header['units'] == ['mV'] * 12,
                    'Source schema mismatch')
            np.testing.assert_allclose(wave[i], original, rtol=1e-7, atol=1e-7)
    require(info['transformations_fitted'] == [] and info['normalization'] == 'none',
            'Unexpected fitted transformation')
    return {'passed': True, 'all_rows_alignment_checked': len(ids),
            'all_rows_labels_reconstructed': len(ids), 'all_waveform_values_finite': True,
            'source_waveform_round_trips': len(source_checks) if check_sources else 0,
            'patient_overlap': False, 'fitted_transformations': []}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--raw-dir', default='data/raw/ptb-xl-1.0.3')
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir, args.raw_dir), indent=2))
