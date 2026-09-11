#!/bin/bash

# UMAP over ~1.5m x 768-dim embeddings is CPU work: pynndescent neighbours,
# spectral init, then transform. GPU umap is not deterministic across runs.
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --mail-user="maxca"
#SBATCH --output=data/logs/reduce_%j.out
#SBATCH --error=data/logs/reduce_%j.err
#SBATCH --chdir=/home/maxca/climate-literature-ipcc
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL

set -e
set -o xtrace

unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export UV_OFFLINE=1

.venv/bin/python -m climate_literature.embed.reduce
