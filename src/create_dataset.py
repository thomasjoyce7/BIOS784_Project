"""Build aligned PTB-XL datasets only after complete raw release verification."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import traceback
import numpy as np
import pandas as pd
import yaml
from construct_labels import CLASSES, MEANINGS, construct_labels, label_summary
from download_ptbxl import VERSION, SOURCE, verify, sha256
from preprocess_metadata import load_metadata, create_splits, predictors, dictionary, validate_record_index
from preprocess_ecg import preprocess_ecg
from validate_dataset import validate

REPO = Path(__file__).resolve().parents[1]
OFFICIAL_COUNTS = dict(zip(CLASSES, [9514, 5469, 5235, 4898, 2649]))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else REPO / p).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/preprocessing.yaml')
    parser.add_argument('--raw-dir', help='Parent of ptb-xl-1.0.3')
    parser.add_argument('--output-dir')
    parser.add_argument('--report-dir')
    parser.add_argument('--limit', type=int, help='Smoke test only; use a separate output/report directory')
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text())
    for key in ['raw_dir', 'output_dir', 'report_dir']:
        override = getattr(args, key)
        if override:
            config[key] = override
    if config['version'] != VERSION or config['sampling_frequency'] != 100 or config['samples'] != 1000:
        raise ValueError('This pipeline supports only PTB-XL 1.0.3 at 100 Hz, 1000 samples')
    if config['leads'] != ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        raise ValueError('Expected official lead order')
    root = resolve(config['raw_dir']) / f'ptb-xl-{VERSION}'
    output, report = resolve(config['output_dir']), resolve(config['report_dir'])
    if args.limit is not None and (args.limit <= 0 or output == resolve('data/processed') or report == resolve('reports/preprocessing_qc')):
        raise ValueError('Smoke tests need a positive limit and separate output AND report directories')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is nonempty; choose a new output directory to avoid overwriting a completed dataset')
    raw_parent = resolve(config['raw_dir'])
    if (output == report or output in report.parents or report in output.parents
            or any(p == raw_parent or raw_parent in p.parents or p in raw_parent.parents
                   for p in [output, report])):
        raise ValueError('Raw, processed, and report locations must be separate, non-nested directories')
    report.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        logging.info('Checking complete release SHA-256 manifest before preprocessing')
        entries, bad = verify(root)
        if bad:
            write_json(report / 'raw_verification_failure.json', {'missing_or_corrupt': bad})
            raise ValueError(f'Incomplete release: {len(bad)} files missing/corrupt')
        all_df = load_metadata(root)
        if len(all_df) != 21799:
            raise ValueError(f'Unexpected 1.0.3 recording count: {len(all_df)}')
        for name in ['filename_lr', 'filename_hr']:
            for record in all_df[name]:
                for suffix in ['.hea', '.dat']:
                    if record + suffix not in entries:
                        raise ValueError(f'Record absent from official manifest: {record}{suffix}')
        index_warnings = validate_record_index(root, all_df)
        mapping = pd.read_csv(root / 'scp_statements.csv', index_col=0)
        if mapping.index.duplicated().any():
            raise ValueError('Duplicate SCP mapping keys')
        all_labels = construct_labels(all_df, mapping)
        if label_summary(all_labels)['positive'] != OFFICIAL_COUNTS:
            raise ValueError('Superclass counts differ from official version 1.0.3 documentation')
        df = all_df.iloc[:args.limit].copy() if args.limit else all_df.copy()
        valid, excluded = preprocess_ecg(df, root, stage, config, report / 'exclusions.csv')
        df = df.set_index('ecg_id').loc[valid].reset_index()
        pred, splits = predictors(df), create_splits(df)
        labels = all_labels.set_index('ecg_id').loc[valid].reset_index()
        for name, table in [('predictors', pred), ('labels', labels), ('splits', splits)]:
            table.to_csv(stage / f'{name}.csv', index=False)
        dictionary(all_df).to_csv(report / 'data_dictionary.csv', index=False)
        pd.DataFrame({'variable': CLASSES, 'meaning': MEANINGS, 'dtype': 'binary integer',
                      'units': '0 absent / 1 present', 'missing_count': 0}).to_csv(report / 'outcome_dictionary.csv', index=False)
        manifest_hash = sha256(root / 'SHA256SUMS.txt')
        info = {'version': VERSION, 'source_url': SOURCE, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'shape': [len(df), 1000, 12], 'dtype': 'float32', 'units': 'mV', 'sampling_frequency': 100,
                'leads': config['leads'], 'recording_order': 'ascending ecg_id; identical across all outputs',
                'label_order': CLASSES, 'normalization': 'none', 'transformations_fitted': [],
                'raw_manifest_sha256': manifest_hash, 'config': config, 'smoke_test': bool(args.limit),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'wfdb', 'PyYAML']},
                'source_sha256': {p.name: sha256(p) for p in sorted((REPO / 'src').glob('*.py'))}}
        write_json(stage / 'dataset.json', info)
        validation = validate(stage, root)
        partitions = {}
        for partition in ['train', 'validation', 'test']:
            mask = splits.partition.eq(partition)
            partitions[partition] = {'recordings': int(mask.sum()),
                                      'patients': int(splits.loc[mask, 'patient_id'].nunique()),
                                      'labels': label_summary(labels.loc[mask])}
        patient_sets = {p: set(splits.loc[splits.partition.eq(p), 'patient_id']) for p in partitions}
        overlaps = {f'{a}/{b}': len(patient_sets[a] & patient_sets[b])
                    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
        qc = {'status': 'PASS', 'smoke_test': bool(args.limit), 'source_version': VERSION,
              'raw_files_verified': len(entries), 'raw_manifest_sha256': manifest_hash,
              'raw_recordings': len(all_df), 'raw_patients': int(all_df.patient_id.nunique()),
              'attempted_recordings': len(valid) + len(excluded), 'processed_recordings': len(valid),
              'processed_patients': int(df.patient_id.nunique()), 'excluded_recordings': len(excluded),
              'exclusions': excluded, 'waveforms': info['shape'], 'dtype': info['dtype'], 'units': 'mV',
              'predictor_missingness': {c: {'count': int(pred[c].isna().sum()), 'fraction': float(pred[c].isna().mean())}
                                       for c in pred.columns if c not in ['ecg_id', 'patient_id']},
              'age_deidentified_count': int(pred.age_deidentified.fillna(False).sum()),
              'labels': label_summary(labels), 'official_full_release_labels': OFFICIAL_COUNTS,
              'partitions': partitions, 'patient_overlap_counts': overlaps, 'validation': validation,
              'warnings': index_warnings + ['Source ages above 89 are deidentified; source values are preserved.',
                           'All-zero label rows are retained; use has_diagnostic_label for diagnostic-task eligibility.']}
        qc['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(stage.iterdir()) if p.is_file()}
        write_json(stage / 'COMPLETE.json', {'status': 'PASS', 'recordings': len(df), 'validation': validation})
        if output.exists():
            output.rmdir()  # Only an empty output directory is permitted.
        stage.rename(output)
        write_json(report / 'qc_report.json', qc)
        lines = ['# PTB-XL preprocessing QC', '', f"Status: PASS ({'smoke subset' if args.limit else 'complete release'}).",
                 f'Raw: {len(entries):,} files verified; {len(all_df):,} recordings; {all_df.patient_id.nunique():,} patients.',
                 f'Processed: {len(valid):,} recordings; {len(excluded)} exclusions; float32 ({len(valid)}, 1000, 12), mV.', '',
                 '| Partition | Recordings | Patients |', '|---|---:|---:|']
        lines += [f"| {p} | {s['recordings']} | {s['patients']} |" for p, s in partitions.items()]
        lines += ['', '| Label | Positive | Fraction |', '|---|---:|---:|']
        lines += [f"| {c} | {qc['labels']['positive'][c]} | {qc['labels']['positive_fraction'][c]:.4f} |" for c in CLASSES]
        lines += ['', f"Multilabel: {qc['labels']['multilabel_recordings']}; no diagnostic label: {qc['labels']['without_diagnostic_label']}.",
                  f'Patient overlap counts: {overlaps}.', 'All serialized rows and labels checked; all waveform values finite; 32 source round trips (or all if fewer).',
                  'No fitted transformations, imputation, filtering, or normalization.', '', 'See qc_report.json for missingness, partition label distributions, output sizes/hashes, and warnings.',
                  'See exclusions.csv for every excluded recording; header only means no exclusions.',
                  '', 'Warnings:'] + ['- ' + warning for warning in qc['warnings']]
        (report / 'qc_report.md').write_text('\n'.join(lines) + '\n')
        (report / 'failure.json').unlink(missing_ok=True)
        logging.info('PASS: %d recordings saved in %s', len(df), output)
    except Exception:
        write_json(report / 'failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
