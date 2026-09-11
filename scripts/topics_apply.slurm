#!/bin/bash
# Apply a chosen run to the whole corpus: reuse the frozen vectorizer, transform
# every paper in row-shards, write doc-topic scores under data/topics/doc_topics.
# Single task; the shard loop keeps peak memory bounded (one shard of sparse
# tf-idf at a time). Set TAG to the run chosen from the comparison sheet.
# Prerequisites: same one-time login-node setup as topics_sweep.slurm.

#SBATCH --job-name=topics-apply
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/maxca/climate-literature-ipcc
#SBATCH --mail-user="maxca"
#SBATCH --mail-type=END,FAIL
#SBATCH --output=data/logs/topics_apply_%j.out
#SBATCH --error=data/logs/topics_apply_%j.err

set -e
set -o xtrace

unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1
export UV_OFFLINE=1
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

TAG="${TAG:-K140_a0.1}"   # override with: TAG=K140_a0.1 sbatch ...
mkdir -p data/logs

uv run --frozen --extra topics python -m climate_literature.topics.train apply --tag "$TAG"

# After it finishes, release the chosen run as a re-applyable artifact
# (login node), tag the code it was built from, and version the data:
#   uv run --frozen --extra topics python -m climate_literature.topics.train finalize --tag "$TAG"
#   git tag "topics-${TAG}-v1" && git push --tags
#   uv run dvc add data/topics && git add data/topics.dvc && git commit
