"""Per-episode metrics, logged under ``Episode_Metrics/`` during training.

Counters come from the fruit pool, which is wiped on reset, so each value is
for the episode that just ended.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .fruit_pool import get_fruit_pool

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "fruits_spawned",
  "fruits_sliced",
  "fruits_missed",
  "slice_rate",
  "blade_speed",
  "fruits_in_air",
]


def fruits_spawned(env: ManagerBasedRlEnv) -> torch.Tensor:
  return get_fruit_pool(env, update=True).spawned_count.float()


def fruits_sliced(env: ManagerBasedRlEnv) -> torch.Tensor:
  return get_fruit_pool(env, update=True).sliced_count.float()


def fruits_missed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return get_fruit_pool(env, update=True).missed_count.float()


def slice_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sliced / (sliced + missed): the share of finished fruit that were cut.
  Fruit still in the air at the end of the episode are not counted."""
  pool = get_fruit_pool(env, update=True)
  done = (pool.sliced_count + pool.missed_count).clamp_min(1)
  return pool.sliced_count / done


def blade_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Blade speed [m/s] over the last step. With reduce="max": the episode's peak
  (compare with MIN_BLADE_SPEED to see whether the policy swings fast enough)."""
  return get_fruit_pool(env, update=True).blade_speed


def fruits_in_air(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fruit currently in the air. With reduce="max": the episode's peak."""
  return get_fruit_pool(env, update=True).active.sum(1).float()
