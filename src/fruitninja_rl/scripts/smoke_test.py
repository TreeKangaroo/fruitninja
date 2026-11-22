"""Sanity-check the slicing env with scripted policies (no learning).

  uv run python -m fruitninja_rl.scripts.smoke_test [--num-envs 64] [--device cpu]

Runs one full episode (a stream of fruit) per policy and checks that:
  1. a "do nothing" policy never slices (every fruit is missed), and
  2. a hand-written "chase the lowest fruit" policy slices some fruit, which
     shows the cut condition is reachable and the reward signal works.
"""

import argparse

import torch

from fruitninja_common.fruit import NUM_FRUIT_SLOTS
from fruitninja_rl.tasks.mdp import get_fruit_pool
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Fruitninja-Slice-Iiwa"


def chase_fruit(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Move the blade point toward where the lowest fruit will be ~0.1 s from now."""
  obs = env.observation_manager.compute()["actor"]
  terms = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  start = 0
  parts = {}
  for name, dim in zip(terms, dims, strict=True):
    parts[name] = obs[:, start : start + dim[0]]
    start += dim[0]
  nearest = parts["upcoming_fruit"][:, :7]  # rel pos (3), vel (3), valid (1)
  lead = nearest[:, 0:3] + 0.1 * nearest[:, 3:6]
  action = torch.zeros(env.num_envs, 6, device=env.device)
  action[:, :3] = (lead / 0.05).clamp(-1.0, 1.0)  # 0.05 m = delta_pos_scale
  return action


def run(env: ManagerBasedRlEnv, policy) -> dict[str, int]:
  """Fruit spawned/sliced/missed over one episode, summed over envs, and the
  most fruit that were ever in the air in one env."""
  env.reset()
  pool = get_fruit_pool(env)
  peak = 0
  # Stop one step before the timeout: the pool is wiped on reset.
  for _ in range(env.max_episode_length - 1):
    env.step(policy(env))
    peak = max(peak, int(pool.active.sum(1).max()))
  return {
    "spawned": int(pool.spawned_count.sum()),
    "sliced": int(pool.sliced_count.sum()),
    "missed": int(pool.missed_count.sum()),
    "peak_in_air": peak,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=args.device)
  print(f"obs dims: { {k: v for k, v in env.observation_manager.group_obs_dim.items()} }")
  print(f"action dim: {env.action_manager.total_action_dim}")
  print(f"episode: {env.max_episode_length} steps ({env.max_episode_length_s} s)")

  def zero(e):
    return torch.zeros(e.num_envs, 6, device=e.device)

  idle = run(env, zero)
  print(f"idle policy:  {idle}")
  assert idle["sliced"] == 0, "Idle robot should never slice."
  assert idle["missed"] > 0, "Fruit should fall past the robot."
  assert idle["spawned"] > 10 * args.num_envs, "Many fruit should drop per episode."
  assert 2 <= idle["peak_in_air"] <= NUM_FRUIT_SLOTS, "Fruit should overlap, up to the cap."

  chase = run(env, chase_fruit)
  done = chase["sliced"] + chase["missed"]
  print(f"chase policy: {chase}  (slice rate {chase['sliced'] / max(done, 1):.0%})")
  assert chase["sliced"] > 0, "Scripted chaser should slice at least one fruit."
  print("OK")
  env.close()


if __name__ == "__main__":
  main()
