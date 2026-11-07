#!/bin/bash
#SBATCH --job-name=ptbxl-preprocess
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm-preprocess-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python src/create_dataset.py --config configs/preprocessing.yaml "$@"
