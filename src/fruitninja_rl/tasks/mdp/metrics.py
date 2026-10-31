"""Per-episode metrics, logged under ``Metrics/`` during training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["slice_success", "closest_approach"]


def slice_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 if the fruit was sliced this step. Use with reduce="max" -> success rate."""
  return env.termination_manager.get_term("fruit_sliced").float()


def closest_approach(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Negative blade-to-fruit distance this step. Use with reduce="max" to log
  minus the episode's closest approach (there is no "min" reduction)."""
  term = env.termination_manager.get_term_cfg("fruit_sliced").func
  return -term.closest_distance
