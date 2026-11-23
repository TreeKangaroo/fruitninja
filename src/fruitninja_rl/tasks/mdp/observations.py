"""Observations. All vectors are in the robot base frame.

The robot base is fixed and unrotated at each env's origin, so a base-frame
vector is a world-frame vector minus the env's base position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .fruit_pool import get_fruit_pool

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ee_pos", "ee_quat", "ee_lin_vel", "upcoming_fruit"]


def _ee_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def ee_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """End-effector (blade point) position. Shape (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  return _ee_pos_w(env, asset_cfg) - robot.data.root_link_pos_w


def ee_quat(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """End-effector orientation (w, x, y, z), sign-fixed so w >= 0. Shape (B, 4)."""
  robot: Entity = env.scene[asset_cfg.name]
  quat = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  return torch.where(quat[:, :1] < 0, -quat, quat)


def ee_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """End-effector linear velocity. Shape (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_lin_vel_w[:, asset_cfg.site_ids].squeeze(1)


def upcoming_fruit(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, num_fruits: int = 4
) -> torch.Tensor:
  """The ``num_fruits`` lowest unsliced fruit in the air, lowest (most urgent) first.
  Use ``num_fruits`` = the pool size to see all of them.

  Per fruit: vector from the end effector (3), velocity (3), valid flag (1).
  Missing fruit (fewer than ``num_fruits`` in the air) are all zeros with
  flag 0. Shape (B, 7 * num_fruits).
  """
  pool = get_fruit_pool(env)
  pos = pool.positions(env)
  vel = pool.velocities(env)
  candidate = pool.active & ~pool.sliced
  order = torch.where(candidate, pos[..., 2], torch.inf).argsort(1)[:, :num_fruits]
  valid = candidate.gather(1, order).float().unsqueeze(-1)  # (B, M, 1)
  idx = order.unsqueeze(-1).expand(-1, -1, 3)
  rel = (pos.gather(1, idx) - _ee_pos_w(env, asset_cfg).unsqueeze(1)) * valid
  v = vel.gather(1, idx) * valid
  return torch.cat([rel, v, valid], dim=-1).flatten(1)
