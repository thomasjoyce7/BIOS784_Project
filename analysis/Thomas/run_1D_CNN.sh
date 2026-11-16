#!/bin/bash
#SBATCH --job-name=ptbxl-1d-cnn
#SBATCH --partition=rtx6k_gpu
#SBATCH --qos=gpu_rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-cnn-%j.out
#SBATCH --error=slurm-cnn-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BIOS784 repository root}"
if [[ ! -f configs/preprocessing.yaml ]]; then
    echo 'Submit this script from the BIOS784 repository root.' >&2
    exit 1
fi
# Existing base environment has torch 2.8.0+cu128 and the notebook/scientific tools.
# For another environment, activate it before sbatch or set CNN_PYTHON to its Python.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ptbxl-cnn-matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
mode="${1:-smoke}"
if [[ "$mode" != smoke && "$mode" != full ]]; then
    echo 'Usage: sbatch analysis/Thomas/run_1D_CNN.sh [smoke|full]' >&2
    exit 2
fi
nvidia-smi
runner_args=(--mode "$mode")
if [[ -n "${CNN_OUTPUT_NOTEBOOK:-}" ]]; then
    runner_args+=(--output-notebook "$CNN_OUTPUT_NOTEBOOK")
fi
"${CNN_PYTHON:-python}" analysis/Thomas/execute_cnn_notebook.py "${runner_args[@]}"
