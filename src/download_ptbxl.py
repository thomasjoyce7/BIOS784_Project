"""Acquire the complete pinned release, then verify every official SHA-256 entry."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import zipfile

VERSION = '1.0.3'
SOURCE = f'https://physionet.org/files/ptb-xl/{VERSION}/'
ZIP_URL = f'https://physionet.org/content/ptb-xl/get-zip/{VERSION}/'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(root):
    entries = {}
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or len(digest) != 64:
            raise ValueError(f'Unsafe or invalid manifest entry: {line}')
        entries[name] = digest
    for name in ['ptbxl_database.csv', 'scp_statements.csv', 'RECORDS', 'LICENSE.txt']:
        if name not in entries:
            raise ValueError(f'Manifest missing {name}')
    for folder in ['records100/', 'records500/']:
        if not any(n.startswith(folder) for n in entries):
            raise ValueError(f'Manifest missing {folder}')
    return entries


def verify(root):
    entries = manifest(root)
    bad = []
    for i, (name, expected) in enumerate(entries.items(), 1):
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            bad.append(name)
        if i % 10000 == 0:
            logging.info('Verified %d/%d files', i, len(entries))
    return entries, bad


def curl(url, destination, resume=False):
    cmd = ['curl', '--fail', '--location', '--retry', '5', '--retry-delay', '3',
           '--connect-timeout', '30', '--output', str(destination)]
    if resume:
        cmd += ['--continue-at', '-']
    subprocess.run(cmd + [url], check=True)


def download(raw_dir, verify_only=False):
    raw_dir = Path(raw_dir)
    root = raw_dir / f'ptb-xl-{VERSION}'
    root.mkdir(parents=True, exist_ok=True)
    checksum = root / 'SHA256SUMS.txt'
    if not checksum.exists():
        if verify_only:
            raise FileNotFoundError(checksum)
        temp = checksum.with_suffix('.part')
        curl(SOURCE + checksum.name, temp)
        temp.replace(checksum)
    entries, bad = verify(root)
    archive = raw_dir / f'ptb-xl-{VERSION}.zip.part'
    if bad and not verify_only and not archive.exists() and len(bad) < len(entries):
        # Repair an existing release without downloading verified waveform files again.
        for i, name in enumerate(bad, 1):
            logging.info('Repairing %d/%d: %s', i, len(bad), name)
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(dest.suffix + '.part')
            curl(SOURCE + name, temp, resume=temp.exists())
            if sha256(temp) != entries[name]:
                raise ValueError(f'Checksum mismatch: {name}; remove {temp} before retrying')
            temp.replace(dest)
        entries, bad = verify(root)
    if bad and not verify_only:
        if not zipfile.is_zipfile(archive):
            curl(ZIP_URL, archive, resume=True)
        # Extract only missing/corrupt files. Never rewrite verified raw files.
        with zipfile.ZipFile(archive) as z:
            lookup = {}
            for info in z.infolist():
                parts = Path(info.filename).parts
                for offset in (0, 1):
                    candidate = '/'.join(parts[offset:])
                    if candidate in entries:
                        lookup[candidate] = info
            for i, name in enumerate(bad, 1):
                if i % 10000 == 0:
                    logging.info("Extracting %d/%d files", i, len(bad))
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if name not in lookup:
                    raise ValueError(f'Archive missing {name}')
                temp = dest.with_suffix(dest.suffix + '.part')
                with z.open(lookup[name]) as source, temp.open('wb') as target:
                    import shutil
                    shutil.copyfileobj(source, target, 1024 * 1024)
                if sha256(temp) != entries[name]:
                    raise ValueError(f'Checksum mismatch: {name}')
                temp.replace(dest)
        entries, bad = verify(root)
        if not bad:
            archive.unlink()
    bad_set = set(bad)
    report = {'version': VERSION, 'source_url': SOURCE, 'archive_url': ZIP_URL,
              'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': sha256(checksum), 'expected_files': len(entries),
              'verified_files': len(entries) - len(bad), 'missing_or_corrupt': bad,
              'verified_bytes': sum((root / n).stat().st_size for n in entries if n not in bad_set)}
    (raw_dir / 'download_status.json').write_text(json.dumps(report, indent=2) + '\n')
    if bad:
        raise RuntimeError(f'{len(bad)} missing/corrupt files; see download_status.json')
    logging.info('Complete release verified: %d files', len(entries))
    return root


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    download(args.raw_dir, args.verify_only)
