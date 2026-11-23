"""Train the slicing policy with PPO. Thin wrapper around mjlab's `train` CLI.

  uv run python -m fruitninja_rl.scripts.train [OPTIONS]
  # equivalent to: uv run train Fruitninja-Slice-Iiwa [OPTIONS]
  # (single-fruit task: Fruitninja-Slice-Iiwa-Single)

Common options:
  --fruit-mode multi|single        10 s fruit stream (default) or one fruit per episode
  --env.scene.num-envs 4096        parallel envs (use thousands on a GPU)
  --agent.max-iterations 2000
  --agent.run-name my_experiment
  --gpu-ids None                   train on CPU (slow; for debugging only)
  --video False                    disable video recording
  --video-interval 2000            env steps between videos (24 steps = 1 iteration)
  --video-length 500               frames per video (10 s at 50 Hz = one episode)
  --agent.logger tensorboard       log locally instead of to wandb
  --agent.run-name my_experiment   wandb run name
  --help                           list every overridable field
"""

import sys

from fruitninja_rl.scripts._task_select import select_task

from mjlab.scripts.train import main as mjlab_train


def main() -> None:
  select_task()
  # Record videos by default (uploaded to wandb by the logger). One video every
  # 2400 env steps = every 100 PPO iterations, matching save_interval.
  # Explicit user flags win.
  for flag, value in (("video", "True"), ("video-interval", "2400"), ("video-length", "500")):
    if not any(a.lstrip("-").replace("_", "-").split("=")[0] == flag for a in sys.argv):
      sys.argv += [f"--{flag}", value]
  mjlab_train()


if __name__ == "__main__":
  main()
