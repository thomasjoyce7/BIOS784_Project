"""Execute the CPU RF notebook, with signature gates and recoverable model caches."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'configs/preprocessing.yaml').is_file())


def implementation_signature(notebook):
    code = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    return hashlib.sha256((code + (HERE / 'rf_pipeline.py').read_text()).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    args = parser.parse_args()
    path = HERE / 'Random_Forest_model.ipynb'
    notebook = nbformat.read(path, as_version=4)
    identity = implementation_signature(notebook)
    if args.mode == 'full':
        gate_path = HERE / 'results/smoke/execution_status.json'
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        if gate.get('status') != 'PASS' or gate.get('implementation_signature') != identity:
            raise RuntimeError('Run the current-code smoke test before full training.')
    os.environ['RF_MODE'] = args.mode
    os.environ.setdefault('RF_RUN_NAME', args.mode)
    run_name = os.environ['RF_RUN_NAME']
    if Path(run_name).name != run_name or run_name in {'.', '..'}:
        raise ValueError('RF_RUN_NAME must be a single folder name.')
    output = HERE / 'results' / run_name
    output.mkdir(parents=True, exist_ok=True)
    lock = (HERE / 'results/.execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = dict(mode=args.mode, status='running', implementation_signature=identity,
                  started_utc=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  test_outcomes_evaluated=args.mode == 'full')
    began = time.perf_counter()
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try:
        if args.mode == 'smoke':
            env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
            tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(HERE / 'test_rf_pipeline.py')],
                                   cwd=ROOT, env=env, text=True, capture_output=True)
            print(tests.stdout, flush=True)
            if tests.returncode:
                print(tests.stderr, file=sys.stderr)
                raise RuntimeError('Random forest correctness tests failed.')
            matches = re.search(r'(\d+) passed', tests.stdout)
            report['correctness_tests_passed'] = int(matches.group(1)) if matches else 0
        client = NotebookClient(notebook, km=manager, timeout=7200, kernel_name='python3',
                                resources={'metadata': {'path': str(HERE)}})
        client.execute()
        nbformat.validate(notebook)
        report.update(status='PASS', code_cells=sum(c.cell_type == 'code' for c in notebook.cells))
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAIL',
                      error=type(exc).__name__)
        raise
    finally:
        try:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        except Exception:
            pass
        report['wall_seconds'] = time.perf_counter() - began
        nbformat.write(notebook, path)
        (output / 'execution_status.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        lock.close()


if __name__ == '__main__':
    main()
