#!/bin/bash
# Fit the NMF topic model over the K x alpha grid on a deterministic sample.
# Single task, CPU, memory-heavy (dense-ish sample tf-idf + nndsvda init).
# NMF's multiplicative solver is BLAS-threaded, so give it cores.
#
# One-time setup on a LOGIN node (internet there, compute nodes are offline):
#   git pull
#   uv sync --frozen --extra topics
#   uv run dvc pull data/predictions      # needs .dvc/config.local
#   uv run --no-project --with nltk scripts/seed_nltk_data.py

#SBATCH --job-name=topics-sweep
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/maxca/climate-literature-ipcc
#SBATCH --mail-user="maxca"
#SBATCH --mail-type=END,FAIL
#SBATCH --output=data/logs/topics_sweep_%j.out
#SBATCH --error=data/logs/topics_sweep_%j.err

set -e
set -o xtrace

unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1
# Let OpenBLAS use the allocated cores for the NMF inner loops.
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p data/logs

uv run --frozen --extra topics python -m climate_literature.topics.train sweep \
  --sample 200000

# Pick K from the comparison sheet (login node or io partition):
#   uv run --frozen --extra topics python -m climate_literature.topics.compare adjacent --alpha 0.1
