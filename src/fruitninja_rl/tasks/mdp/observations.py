"""Observations. All vectors are in the robot base frame.

The robot base is fixed and unrotated at each env's origin, so a base-frame
vector is a world-frame vector minus the env's base position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ee_pos", "ee_quat", "ee_lin_vel", "fruit_pos_rel_ee", "fruit_lin_vel"]


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


def fruit_pos_rel_ee(
  env: ManagerBasedRlEnv, fruit_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Vector from the end effector to the fruit center. Shape (B, 3)."""
  fruit: Entity = env.scene[fruit_name]
  return fruit.data.root_link_pos_w - _ee_pos_w(env, asset_cfg)


def fruit_lin_vel(env: ManagerBasedRlEnv, fruit_name: str) -> torch.Tensor:
  """Fruit linear velocity. Shape (B, 3)."""
  fruit: Entity = env.scene[fruit_name]
  return fruit.data.root_link_lin_vel_w
