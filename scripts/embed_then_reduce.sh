#!/usr/bin/env bash
# Submit the embedding pipeline in one go: the GPU embedding array first,
# then the CPU UMAP job, held until every array task ends OK (afterok — if
# any embedding task fails, the reduce never runs and the fix is just to
# resume the array and resubmit the reduce).
#
#   ./scripts/embed_then_reduce.sh
#   NUM_EMBED_TASKS=16 ./scripts/embed_then_reduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

last_task=$(( ${NUM_EMBED_TASKS:-8} - 1 ))

embed_id=$(sbatch --parsable --array="0-${last_task}" scripts/embed.slurm)
echo "embedding array submitted: ${embed_id}_0-${last_task}"

reduce_id=$(sbatch --parsable --dependency="afterok:${embed_id}_[0-${last_task}]" scripts/reduce.slurm)
echo "reduce queued: ${reduce_id} (waits for ${embed_id}_0-${last_task})"
echo "watch with: squeue -u \$USER"
