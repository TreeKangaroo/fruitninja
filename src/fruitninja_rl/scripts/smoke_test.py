"""Sanity-check the slicing env with scripted policies (no learning).

  uv run python -m fruitninja_rl.scripts.smoke_test [--num-envs 64] [--device cpu]

Checks that:
  1. a "do nothing" policy never slices (every fruit is missed), and
  2. a hand-written "chase the fruit" policy does slice some fruit, which
     shows the cut condition is reachable and the reward signal works.
"""

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Fruitninja-Slice-Iiwa"


def chase_fruit(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Move the blade point toward where the fruit will be ~0.1 s from now."""
  obs = env.observation_manager.compute()["actor"]
  terms = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  start = 0
  parts = {}
  for name, dim in zip(terms, dims, strict=True):
    parts[name] = obs[:, start : start + dim[0]]
    start += dim[0]
  lead = parts["fruit_pos_rel_ee"] + 0.1 * parts["fruit_lin_vel"]
  action = torch.zeros(env.num_envs, 6, device=env.device)
  action[:, :3] = (lead / 0.05).clamp(-1.0, 1.0)  # 0.05 m = delta_pos_scale
  return action


def run(env: ManagerBasedRlEnv, policy, num_steps: int) -> dict[str, int]:
  counts = {"fruit_sliced": 0, "fruit_missed": 0, "time_out": 0}
  env.reset()
  for _ in range(num_steps):
    env.step(policy(env))
    for name in counts:
      counts[name] += int(env.termination_manager.get_term(name).sum())
  return counts


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--steps", type=int, default=100)
  args = parser.parse_args()

  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=args.device)
  print(f"obs dims: { {k: v for k, v in env.observation_manager.group_obs_dim.items()} }")
  print(f"action dim: {env.action_manager.total_action_dim}")

  def zero(e):
    return torch.zeros(e.num_envs, 6, device=e.device)

  idle = run(env, zero, args.steps)
  print(f"idle policy:  {idle}")
  assert idle["fruit_sliced"] == 0, "Idle robot should never slice."
  assert idle["fruit_missed"] > 0, "Fruit should fall past the robot."

  chase = run(env, chase_fruit, args.steps)
  total = chase["fruit_sliced"] + chase["fruit_missed"]
  print(f"chase policy: {chase}  (slice rate {chase['fruit_sliced'] / max(total, 1):.0%})")
  assert chase["fruit_sliced"] > 0, "Scripted chaser should slice at least one fruit."
  print("OK")
  env.close()


if __name__ == "__main__":
  main()
