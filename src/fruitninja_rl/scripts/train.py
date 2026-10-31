"""Train the slicing policy with PPO. Thin wrapper around mjlab's `train` CLI.

  uv run python -m fruitninja_rl.scripts.train [OPTIONS]
  # equivalent to: uv run train Fruitninja-Slice-Iiwa [OPTIONS]

Common options:
  --env.scene.num-envs 4096        parallel envs (use thousands on a GPU)
  --agent.max-iterations 2000
  --agent.run-name my_experiment
  --gpu-ids None                   train on CPU (slow; for debugging only)
  --help                           list every overridable field
"""

import sys

from mjlab.scripts.train import main as mjlab_train

DEFAULT_TASK = "Fruitninja-Slice-Iiwa"


def main() -> None:
  if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
    sys.argv.insert(1, DEFAULT_TASK)
  mjlab_train()


if __name__ == "__main__":
  main()
