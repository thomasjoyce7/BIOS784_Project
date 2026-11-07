#!/bin/bash
#SBATCH --job-name=ptbxl-tests
#SBATCH --partition=general
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-tests-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest -q tests
