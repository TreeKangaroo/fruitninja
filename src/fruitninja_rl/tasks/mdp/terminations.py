"""Terminations.

``fruit_sliced`` is where slicing is detected. Rewards and metrics read its
result via ``env.termination_manager.get_term("fruit_sliced")`` rather than
recomputing it. (mjlab computes terminations before rewards each step.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fruitninja_common import slicing
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

__all__ = ["fruit_sliced"]


class fruit_sliced:
  """True on the step where the blade slices the fruit (see ``slicing.check_slice``).

  Stateful: the cut check needs blade and fruit positions at the previous step
  so it can detect pass-throughs that happen between two control steps.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    n, device = env.num_envs, env.device
    self.prev_blade_pos = torch.zeros(n, 3, device=device)
    self.prev_fruit_pos = torch.zeros(n, 3, device=device)
    # False right after a reset, when there is no previous step to compare to.
    self.has_prev = torch.zeros(n, dtype=torch.bool, device=device)
    # Diagnostics from the latest step (handy for debugging and plotting).
    self.closest_distance = torch.zeros(n, device=device)
    self.blade_speed = torch.zeros(n, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.has_prev[slice(None) if env_ids is None else env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    fruit_name: str,
    asset_cfg: SceneEntityCfg,
    hit_radius: float,
    min_blade_speed: float,
  ) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    fruit: Entity = env.scene[fruit_name]
    blade_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    fruit_pos = fruit.data.root_link_pos_w

    sliced, self.closest_distance, self.blade_speed = slicing.check_slice(
      self.prev_blade_pos,
      blade_pos,
      self.prev_fruit_pos,
      fruit_pos,
      dt=env.step_dt,
      hit_radius=hit_radius,
      min_blade_speed=min_blade_speed,
    )
    sliced = sliced & self.has_prev

    self.prev_blade_pos[:] = blade_pos
    self.prev_fruit_pos[:] = fruit_pos
    self.has_prev[:] = True
    return sliced
