#!/bin/bash
# Slurm job: serve the viser viewer (scripts/play.py) from a GPU compute node.
# This only visualizes. It does not train.
#
# Run everything from the repo root, on the cluster:
#   uv sync --frozen --python 3.12    # once, on the login node (needs internet)
#   mkdir -p logs/slurm               # Slurm needs this before the job starts
#   sbatch src/fruitninja_rl/scripts/play.sbatch            # zero-action arm
#   cat logs/slurm/fruitninja-play_<jobid>.out              # prints the ssh tunnel
#   scancel <jobid>                   # when done: the viewer never exits by itself
#
# Arguments after the script go to play.py:
#   sbatch src/fruitninja_rl/scripts/play.sbatch --checkpoint-file <path/to/model.pt>
#   sbatch src/fruitninja_rl/scripts/play.sbatch --fruit-mode single
#
# Slurm options go before the script, e.g. to use another partition:
#   sbatch --partition=<name> src/fruitninja_rl/scripts/play.sbatch
#
# Without sbatch (stops on Ctrl-C):
#   srun --partition=mit_normal_gpu --gpus=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty \
#       bash src/fruitninja_rl/scripts/play.sbatch

#SBATCH --job-name=fruitninja-play
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "not in a Slurm job. Start this with sbatch or srun, not on the login node."
cd "${SLURM_SUBMIT_DIR}"
[[ -f pyproject.toml ]] || die "submit from the repo root (submitted from ${PWD})."

# Batch jobs don't load your shell profile, so add uv's default install dirs.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || die "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
[[ -x .venv/bin/python ]] || die "no .venv. On the login node run: uv sync --frozen --python 3.12"

# Compute nodes often have no internet, so never install from inside the job.
export UV_NO_SYNC=1
# Otherwise output sits in a buffer and the log looks empty.
export PYTHONUNBUFFERED=1
# The browser does the rendering. mjlab would otherwise load EGL on Linux, which
# fails on nodes without it.
export MUJOCO_GL="${MUJOCO_GL:-disable}"

# Fail loudly, because play.py would silently fall back to the CPU.
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
uv run python -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' ||
  die "torch cannot use a GPU. Either none was allocated, or the driver is too old (uv.lock has CUDA 13, which needs driver 580+)."

# viser quietly moves to the next port if its port is taken, which would point
# the tunnel at someone else's process. So pick a free port first.
PORT="$(uv run python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export _VISER_PORT_OVERRIDE="$PORT"
NODE="${SLURMD_NODENAME:-$(hostname -s)}"

cat <<EOF
==========================================================================
Viewer is starting on ${NODE}:${PORT} (the first start takes a few minutes).

1. On your LAPTOP, run this and leave it open:

     ssh -N -L 8080:${NODE}:${PORT} ${USER}@orcd-login.mit.edu

2. When "viser" appears below, open http://localhost:8080
   (ignore the address that viser prints)

3. When done, on the cluster: scancel ${SLURM_JOB_ID}

Note: viser has no password. Others on the cluster network can open this port.
==========================================================================
EOF

exec uv run python -m fruitninja_rl.scripts.play --viewer viser --device cuda:0 "$@"
