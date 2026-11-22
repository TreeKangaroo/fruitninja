"""Terminations specific to the slicing task.

The multi-fruit task ends on timeout only. The single-fruit task also ends as
soon as its fruit has been resolved (sliced, or fell to the floor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .fruit_pool import get_fruit_pool

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["fruits_resolved"]


def fruits_resolved(env: ManagerBasedRlEnv, num_fruits: int = 1) -> torch.Tensor:
  """True once ``num_fruits`` fruit have each been sliced or have hit the floor."""
  pool = get_fruit_pool(env, update=True)
  return (pool.sliced_count + pool.missed_count) >= num_fruits
