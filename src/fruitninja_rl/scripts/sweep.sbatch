#!/bin/bash
# Slurm job array: one training run per sweep grid point, one GPU each.
#
#   uv run python -m fruitninja_rl.scripts.sweep --list   # count the runs
#   mkdir -p logs/slurm                                   # Slurm needs this first
#   sbatch src/fruitninja_rl/scripts/sweep.sbatch         # from the repo root
#   squeue -u $USER                                       # check progress
#   uv run python -m fruitninja_rl.scripts.sweep --collect  # rank when done
#
# Set --array to 0-(N-1) for the N runs that --list reports; "%8" caps how
# many run at once. Adjust partition/time/modules for your cluster.

#SBATCH --job-name=fruitninja-sweep
#SBATCH --array=0-23%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
# #SBATCH --partition=<gpu-partition>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# module load cuda   # if your cluster needs it

# Batch jobs don't load your shell profile, so add uv's default install dirs.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "uv not found. Install it on the cluster first:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
uv run python -m fruitninja_rl.scripts.sweep --index "${SLURM_ARRAY_TASK_ID}"
