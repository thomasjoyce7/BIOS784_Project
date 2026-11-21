#!/bin/bash
#SBATCH --job-name=ptbxl-random-forest
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-rf-%j.out
#SBATCH --error=slurm-rf-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BIOS784 repository root}"
if [[ ! -f configs/preprocessing.yaml ]]; then
    echo 'Submit from the BIOS784 repository root.' >&2
    exit 1
fi
# Parallelism is within one forest's trees; labels and trials run sequentially.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ptbxl-rf-matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
mode="${1:-smoke}"
if [[ "$mode" != smoke && "$mode" != full ]]; then
    echo 'Usage: sbatch analysis/Thomas/Random_Forest/run_random_forest.sh [smoke|full]' >&2
    exit 2
fi
"${RF_PYTHON:-python}" analysis/Thomas/Random_Forest/execute_rf_notebook.py --mode "$mode"
