"""Rewards. mjlab multiplies every term by ``weight * step_dt``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["termination_term", "ee_fruit_proximity", "joint_vel_limits"]


def termination_term(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  """1.0 on the step where the given termination term fired (e.g. a slice)."""
  return env.termination_manager.get_term(term_name).float()


def ee_fruit_proximity(
  env: ManagerBasedRlEnv, std: float, fruit_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Dense shaping: exp(-d^2 / std^2), d = end-effector-to-fruit distance."""
  robot: Entity = env.scene[asset_cfg.name]
  fruit: Entity = env.scene[fruit_name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  dist_sq = (fruit.data.root_link_pos_w - ee_pos).square().sum(-1)
  return torch.exp(-dist_sq / std**2)


def joint_vel_limits(
  env: ManagerBasedRlEnv, limits: dict[str, float], asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Sum of squared joint-velocity excess over per-joint hardware limits.

  MuJoCo does not enforce velocity limits, so this keeps motions within what
  the real arm can do. ``limits`` maps joint name -> max |velocity| [rad/s].
  """
  robot: Entity = env.scene[asset_cfg.name]
  max_vel = torch.tensor([limits[n] for n in robot.joint_names], device=env.device)
  excess = (robot.data.joint_vel.abs() - max_vel).clamp_min(0.0)
  return excess.square().sum(-1)
