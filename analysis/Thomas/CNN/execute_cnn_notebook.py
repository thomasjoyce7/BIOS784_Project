"""Execute the CNN notebook with the current Python kernel; preserve partial outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'configs/preprocessing.yaml').is_file())


def implementation_signature(notebook):
    code = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    support = (HERE / 'cnn_notebook_support.py').read_text()
    return hashlib.sha256((code + (HERE / 'cnn_pipeline.py').read_text() + support).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--output-notebook', help='Save executed outputs to another notebook; paths resolve from repository root')
    args = parser.parse_args()
    path = HERE / '1D_CNN_model.ipynb'
    output_path = (ROOT / args.output_notebook).resolve() if args.output_notebook else path
    notebook = nbformat.read(path, as_version=4)
    identity = implementation_signature(notebook)
    if args.mode == 'full':
        gate_path = HERE / 'cnn_results/smoke/smoke_review.json'
        if not gate_path.exists():
            raise RuntimeError('Run the default smoke job and review its results before launching full training.')
        gate = json.loads(gate_path.read_text())
        if gate.get('status') != 'PASS' or gate.get('implementation_signature') != identity:
            raise RuntimeError('Full-run gate requires a passed smoke test of the current code.')
    os.environ['CNN_MODE'] = args.mode
    os.environ.setdefault('CNN_RUN_NAME', args.mode)
    output = HERE / 'cnn_results' / os.environ['CNN_RUN_NAME']
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    started = time.perf_counter()
    report = dict(mode=args.mode, status='running', implementation_signature=identity,
                  started_utc=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Execution interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    manager = KernelManager(kernel_name='python3')
    # Match the activated environment rather than an unrelated registered Python kernel.
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try:
        client = NotebookClient(notebook, km=manager, timeout=14400, kernel_name='python3',
                                resources={'metadata': {'path': str(HERE)}})
        client.execute()
        nbformat.validate(notebook)
        report.update(status='PASS', code_cells=sum(c.cell_type == 'code' for c in notebook.cells))
        if args.mode == 'smoke':
            checks = json.loads((output / 'smoke_checks.json').read_text())
            if checks['status'] != 'PASS' or not all(checks['checks'].values()) or checks['test_outcomes_evaluated']:
                raise RuntimeError('Smoke checklist failed.')
            review = dict(report, checks=checks['checks'], test_outcomes_evaluated=False)
            (output / 'smoke_review.json').write_text(json.dumps(review, indent=2) + '\n')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'FAIL',
                      error=type(exc).__name__)
        raise
    finally:
        try:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        except Exception:
            pass  # Preserve the original execution exception, if any.
        report['wall_seconds'] = time.perf_counter() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output_path)
        (output / 'execution_status.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
