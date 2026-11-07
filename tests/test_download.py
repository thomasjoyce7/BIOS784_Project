import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from download_ptbxl import manifest, verify


def test_detects_missing_and_corrupt_files(tmp_path):
    names = ['ptbxl_database.csv', 'scp_statements.csv', 'RECORDS', 'LICENSE.txt',
             'records100/a.dat', 'records500/a.dat']
    digest = hashlib.sha256(b'original').hexdigest()
    (tmp_path / 'SHA256SUMS.txt').write_text(''.join(f'{digest} {n}\n' for n in names))
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
    assert verify(tmp_path)[1] == []
    (tmp_path / names[0]).write_bytes(b'corrupt')
    (tmp_path / names[1]).unlink()
    assert set(verify(tmp_path)[1]) == set(names[:2])


def test_rejects_unsafe_manifest(tmp_path):
    import pytest
    (tmp_path / 'SHA256SUMS.txt').write_text('0' * 64 + ' ../outside\n')
    with pytest.raises(ValueError, match='Unsafe'):
        manifest(tmp_path)


def test_repair_download_and_skip_verified_files(tmp_path, monkeypatch):
    import json
    import zipfile
    import download_ptbxl as module
    root = tmp_path / 'ptb-xl-1.0.3'
    root.mkdir()
    names = ['ptbxl_database.csv', 'scp_statements.csv', 'RECORDS', 'LICENSE.txt',
             'records100/a.dat', 'records500/a.dat']
    digest = hashlib.sha256(b'original').hexdigest()
    (root / 'SHA256SUMS.txt').write_text(''.join(f'{digest} {n}\n' for n in names))
    preserved = root / 'LICENSE.txt'
    preserved.write_bytes(b'original')
    original_mtime = preserved.stat().st_mtime_ns
    (root / 'ptbxl_database.csv').write_bytes(b'corrupt')
    with zipfile.ZipFile(tmp_path / 'ptb-xl-1.0.3.zip.part', 'w') as archive:
        for name in names:
            archive.writestr('ptb-xl-1.0.3/' + name, b'original')
    calls = []
    def fetch(url, destination, resume=False):
        calls.append((url, resume))
        with zipfile.ZipFile(destination, 'w') as archive:
            for name in names:
                archive.writestr('ptb-xl-1.0.3/' + name, b'original')
    monkeypatch.setattr(module, 'curl', fetch)
    assert module.download(tmp_path) == root
    assert calls == []
    assert preserved.stat().st_mtime_ns == original_mtime
    assert module.verify(root)[1] == []
    assert not (tmp_path / 'ptb-xl-1.0.3.zip.part').exists()
    status = json.loads((tmp_path / 'download_status.json').read_text())
    assert status['verified_files'] == len(names) and not status['missing_or_corrupt']
    calls.clear()
    module.download(tmp_path)
    assert calls == []


def test_existing_release_repairs_only_bad_files(tmp_path, monkeypatch):
    import download_ptbxl as module
    root = tmp_path / 'ptb-xl-1.0.3'
    root.mkdir()
    names = ['ptbxl_database.csv', 'scp_statements.csv', 'RECORDS', 'LICENSE.txt',
             'records100/a.dat', 'records500/a.dat']
    digest = hashlib.sha256(b'original').hexdigest()
    (root / 'SHA256SUMS.txt').write_text(''.join(f'{digest} {n}\n' for n in names))
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
    (root / 'records100/a.dat').write_bytes(b'corrupt')
    calls = []
    def fetch(url, destination, resume=False):
        calls.append(url)
        destination.write_bytes(b'original')
    monkeypatch.setattr(module, 'curl', fetch)
    module.download(tmp_path)
    assert calls == [module.SOURCE + 'records100/a.dat']
