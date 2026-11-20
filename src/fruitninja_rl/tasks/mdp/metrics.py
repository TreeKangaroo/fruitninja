"""Per-episode metrics, logged under ``Metrics/`` during training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["slice_success", "closest_approach", "blade_speed", "fruit_height"]


def slice_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 if the fruit was sliced this step. Use with reduce="max" -> success rate."""
  return env.termination_manager.get_term("fruit_sliced").float()


def closest_approach(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Negative blade-to-fruit distance this step. Use with reduce="max" to log
  minus the episode's closest approach (there is no "min" reduction)."""
  term = env.termination_manager.get_term_cfg("fruit_sliced").func
  return -term.closest_distance


def blade_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Blade speed [m/s] over the last step. With reduce="max": the episode's peak
  (compare with MIN_BLADE_SPEED to see whether the policy swings fast enough)."""
  term = env.termination_manager.get_term_cfg("fruit_sliced").func
  return term.blade_speed


def fruit_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Negative fruit height [m] at the end of the step. With reduce="max": minus
  the episode's lowest fruit height, i.e. how far the fruit fell before the episode ended."""
  return -env.scene["fruit"].data.root_link_pos_w[:, 2]
