#!/bin/bash
# Extend the K grid above the 2019 ceiling: the published model ran K=140
# near the top of the old 80..150 window, on 3.6x fewer docs. This sweep fits
# K=160..220 at alpha=0 on the same deterministic sample (SEED, same
# vectorizer), so the ladder and coherence tail can be extended past 150
# before K is chosen. alpha=0 only: the grid showed alpha inert, and these
# runs test range, not regularisation.
#
# Requires the same one-time bootstrap as topics_sweep.slurm (see its header).

#SBATCH --job-name=topics-sweep-high
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/maxca/climate-literature-ipcc
#SBATCH --mail-user="maxca"
#SBATCH --mail-type=END,FAIL
#SBATCH --output=data/logs/topics_sweep_high_%j.out
#SBATCH --error=data/logs/topics_sweep_high_%j.err

set -e
set -o xtrace

unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p data/logs

uv run --frozen --extra topics python -m climate_literature.topics.train sweep \
  --sample 200000 --k 160,180,200,220 --alpha 0.0
