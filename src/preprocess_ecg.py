"""Stream physical WFDB samples to a memory-mapped float32 array."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb


def preprocess_ecg(df, root, output, config, exclusion_path):
    shape = (len(df), config['samples'], len(config['leads']))
    temp = output / 'waveforms.work.npy'
    signals = np.lib.format.open_memmap(temp, mode='w+', dtype='float32', shape=shape)
    valid, excluded = [], []
    for row in df.itertuples():
        try:
            path = Path(row.filename_lr)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe source waveform path')
            wave, fields = wfdb.rdsamp(str(root / path))
            if fields['fs'] != config['sampling_frequency']:
                raise ValueError(f"Sampling frequency {fields['fs']}")
            if wave.shape != shape[1:]:
                raise ValueError(f'Waveform shape {wave.shape}')
            if fields['sig_name'] != config['leads']:
                raise ValueError(f"Lead order {fields['sig_name']}")
            if fields['units'] != ['mV'] * shape[2]:
                raise ValueError(f"Unexpected units {fields['units']}")
            if not np.isfinite(wave).all():
                raise ValueError('Nonfinite waveform values')
            signals[len(valid)] = wave
            valid.append(row.ecg_id)
        except (ValueError, OSError, EOFError) as exc:
            logging.error('Excluding ECG %s: %s', row.ecg_id, exc)
            excluded.append({'ecg_id': row.ecg_id, 'reason': f'{type(exc).__name__}: {exc}'})
        if (len(valid) + len(excluded)) % 1000 == 0:
            logging.info('Read %d/%d ECGs (%d exclusions)', len(valid) + len(excluded), len(df), len(excluded))
    pd.DataFrame(excluded, columns=['ecg_id', 'reason']).to_csv(exclusion_path, index=False)
    signals.flush()
    if not valid:
        raise ValueError('No valid waveforms; see exclusions.csv')
    if len(valid) < len(df):
        final = np.lib.format.open_memmap(output / 'waveforms.npy', mode='w+', dtype='float32',
                                         shape=(len(valid), *shape[1:]))
        for start in range(0, len(valid), 128):
            stop = min(start + 128, len(valid))
            final[start:stop] = signals[start:stop]
        final.flush()
        del final, signals
        temp.unlink()
    else:
        del signals
        temp.rename(output / 'waveforms.npy')
    np.save(output / 'recording_ids.npy', np.asarray(valid, dtype='int64'), allow_pickle=False)
    return valid, excluded
