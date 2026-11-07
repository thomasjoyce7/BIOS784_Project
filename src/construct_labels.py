"""Official SCP diagnostic superclass aggregation, with no likelihood threshold."""
import ast
import pandas as pd

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
MEANINGS = ['Normal ECG', 'Myocardial infarction', 'ST/T changes', 'Conduction disturbance', 'Hypertrophy']


def construct_labels(df, mapping):
    diagnostic = mapping.loc[mapping.diagnostic.eq(1), 'diagnostic_class']
    if diagnostic.isna().any() or not diagnostic.isin(CLASSES).all():
        raise ValueError('Unexpected diagnostic superclass mapping')
    rows = []
    for row in df.itertuples():
        codes = ast.literal_eval(row.scp_codes)
        if not isinstance(codes, dict) or not all(isinstance(c, str) for c in codes):
            raise ValueError(f'Invalid SCP dictionary: ECG {row.ecg_id}')
        unknown = set(codes) - set(mapping.index)
        if unknown:
            raise ValueError(f'Unmapped SCP codes at ECG {row.ecg_id}: {unknown}')
        superclasses = {diagnostic[c] for c in codes if c in diagnostic.index}
        rows.append([row.ecg_id] + [int(c in superclasses) for c in CLASSES])
    out = pd.DataFrame(rows, columns=['ecg_id'] + CLASSES)
    out['has_diagnostic_label'] = out[CLASSES].sum(axis=1).gt(0).astype('uint8')
    return out


def label_summary(labels):
    n = len(labels)
    return {'recordings': n, 'positive': {c: int(labels[c].sum()) for c in CLASSES},
            'positive_fraction': {c: float(labels[c].mean()) if n else 0.0 for c in CLASSES},
            'multilabel_recordings': int(labels[CLASSES].sum(axis=1).gt(1).sum()),
            'without_diagnostic_label': int(labels[CLASSES].sum(axis=1).eq(0).sum()),
            'eligible_recordings': int(labels[CLASSES].sum(axis=1).gt(0).sum())}
