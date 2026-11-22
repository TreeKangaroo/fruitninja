"""Visualize the slicing env. Thin wrapper around mjlab's `play` CLI.

  # No policy: the robot holds its pose while fruit drops (zero action =
  # "keep the end effector where it is"). Good for checking the environment.
  uv run python -m fruitninja_rl.scripts.play --viewer viser

  # A trained policy:
  uv run python -m fruitninja_rl.scripts.play --viewer viser \
      --checkpoint-file logs/rsl_rl/iiwa_slice/<run>/model_<N>.pt

Common options:
  --fruit-mode multi|single  10 s fruit stream (default) or one fruit per episode
  --viewer viser          browser viewer at http://localhost:8080 (works over SSH/WSL)
  --viewer native         MuJoCo desktop viewer
  --agent zero|random     run without a checkpoint (zero is the default if no
                          checkpoint is given)
  --num-envs 4
"""

import sys

from fruitninja_rl.scripts._task_select import select_task

from mjlab.scripts.play import main as mjlab_play

# Any of these flags means the user chose a policy explicitly.
_POLICY_FLAGS = ("--agent", "--checkpoint-file", "--wandb-run-path")


def main() -> None:
  select_task()
  if not any(arg.split("=")[0] in _POLICY_FLAGS for arg in sys.argv):
    print("[INFO] No checkpoint given: running the zero-action agent.")
    sys.argv += ["--agent", "zero"]
  mjlab_play()


if __name__ == "__main__":
  main()
