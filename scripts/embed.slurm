#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --mail-user="maxca"
#SBATCH --output=data/logs/embed_%j.out
#SBATCH --error=data/logs/embed_%j.err
#SBATCH --chdir=/home/maxca/climate-literature-ipcc
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7

set -e
set -o xtrace

module load cuda

unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1
export HF_HOME=/data/rd5/ecs/huggingface-cache


JOB_ID=$SLURM_ARRAY_TASK_ID
NUM_JOBS=$SLURM_ARRAY_TASK_COUNT

.venv/bin/python -m climate_literature.embed.embed --job-id=$JOB_ID --num-jobs=$NUM_JOBS --batch-size=256
