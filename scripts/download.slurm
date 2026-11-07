#!/bin/bash
#SBATCH --job-name=ptbxl-download
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-download-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
python src/download_ptbxl.py "$@"
