#!/bin/bash
# Slurm job for MIT ORCD Engaging: train the slicing policy on one GPU
# (4096 envs, 3000 iterations, every trajectory in fruit.TRAJECTORIES).
#
# Run everything from the repo root, on a login node (orcd-login.mit.edu):
#   uv sync --frozen --python 3.12    # once
#   uv run --no-sync wandb login      # once
#   mkdir -p logs/slurm               # Slurm needs this before the job starts
#   sbatch src/fruitninja_rl/scripts/train.sbatch
#   squeue --me                                       # PD = waiting, R = running
#   tail -f logs/slurm/fruitninja-train_<jobid>.out   # progress and ETA
#
# Arguments after the script go to train.py and override the defaults below:
#   sbatch --time=00:30:00 src/fruitninja_rl/scripts/train.sbatch \
#       --agent.max-iterations 25 --agent.run-name quicktest     # test run
#   sbatch src/fruitninja_rl/scripts/train.sbatch --agent.run-name straight \
#       --env.events.fruit-pool.params.trajectories.xvel.weight 0
#
# mit_normal_gpu stops jobs after 6 h. A checkpoint is saved every 100 iterations.
# To continue a stopped run from its latest checkpoint (max-iterations is the
# number of ADDITIONAL iterations, so 3000 minus the checkpoint's number):
#   sbatch src/fruitninja_rl/scripts/train.sbatch --agent.resume True \
#       --agent.load-run <run folder name> --agent.load-checkpoint 'model_.*' \
#       --agent.max-iterations <remaining>

#SBATCH --job-name=fruitninja-train
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}
warn() {
  echo "WARNING: $*" >&2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "not in a Slurm job. Start this with sbatch, not on the login node."
cd "${SLURM_SUBMIT_DIR}"
[[ -f pyproject.toml ]] || die "submit from the repo root (submitted from ${PWD})."

# Batch jobs don't load your shell profile, so add uv's default install dirs.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || die "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
[[ -x .venv/bin/python ]] || die "no .venv. On the login node run: uv sync --frozen --python 3.12"

# Install on the login node, never from inside the job.
export UV_NO_SYNC=1
export PYTHONUNBUFFERED=1

echo "job ${SLURM_JOB_ID} on $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
uv run python -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' ||
  die "torch cannot use a GPU. Either none was allocated, or the driver is too old (uv.lock has CUDA 13, which needs driver 580+)."

defaults=(
  --env.scene.num-envs 4096
  --agent.max-iterations 3000
  --agent.run-name mixed_concurrent
)

# Videos need headless rendering. Render one test frame, because otherwise a
# node without working EGL would kill the whole run.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if ! MUJOCO_EGL_DEVICE_ID=0 timeout 120 uv run python -c '
import mujoco
model = mujoco.MjModel.from_xml_string("<mujoco/>")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 64, 64)
renderer.update_scene(data)
renderer.render()
renderer.close()'; then
  warn "headless rendering failed on this node: training WITHOUT videos."
  export MUJOCO_GL=disable
  defaults+=(--video False)
fi

# A batch job cannot answer wandb's login prompt, so log offline if wandb is unusable.
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "${NETRC:-$HOME/.netrc}"; then
    warn "no wandb login found (run: uv run --no-sync wandb login)."
    export WANDB_MODE=offline
  elif ! uv run python -c 'import urllib.request; urllib.request.urlopen("https://api.wandb.ai/healthz", timeout=15)'; then
    warn "this node cannot reach wandb."
    export WANDB_MODE=offline
  fi
fi
if [[ "${WANDB_MODE:-}" == offline ]]; then
  warn "logging offline. Afterwards, on the login node: uv run --no-sync wandb sync wandb/offline-run-*"
fi

# "$@" comes last, so your arguments win over the defaults.
exec uv run python -m fruitninja_rl.scripts.train "${defaults[@]}" "$@"
