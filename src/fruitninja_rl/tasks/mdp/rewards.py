"""Rewards. mjlab multiplies every term by ``weight * step_dt``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fruitninja_common import slicing
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "termination_term",
  "ee_fruit_proximity",
  "slice_object_reward",
  "joint_vel_limits",
]


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


def slice_object_reward(
  env: ManagerBasedRlEnv,
  miss_std: float,
  fruit_name: str,
  asset_cfg: SceneEntityCfg,
  epsilon: float = slicing.HIT_RADIUS,
  min_blade_speed: float = slicing.MIN_BLADE_SPEED,
  min_alignment: float = 0.7, # alignment along the blade axes
  blade_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
  """Shaped reward for slicing through the spherical fruit during this step.

  The cut geometry comes from ``slicing.check_slice`` (the same test used by
  the ``fruit_sliced`` termination). The start of the step is reconstructed
  from the current positions and velocities (p_prev = p - v * step_dt).
  Three criteria are scored:

  * Alignment: cosine between the blade axis (``blade_axis`` in the EE site
    frame, rotated to world) and the unit vector from the EE to the fruit
    center. Treated as 1 once the EE is inside the epsilon ball, where the
    direction to the center is ill-defined and the cut is already happening.
  * Interception: the closest EE-to-fruit-center distance d over the step
    must be within ``epsilon``. Shaped as exp(-max(d - epsilon, 0)^2 /
    miss_std^2) so near-misses still get signal.
  * Blade speed: |v_ee| / min_blade_speed, clipped to [0, 1], so slow touches
    (or fruit falling onto a stationary blade) are not rewarded as cuts.

  Returns ``dense + bonus`` in [0, 2]: ``dense`` is the product of the three
  soft terms; ``bonus`` is 1.0 when ``check_slice`` reports a cut (d <=
  epsilon and blade speed >= min_blade_speed) and alignment >=
  ``min_alignment``.
  """
  robot: Entity = env.scene[asset_cfg.name]
  fruit: Entity = env.scene[fruit_name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  ee_quat = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel = robot.data.site_lin_vel_w[:, asset_cfg.site_ids].squeeze(1)
  fruit_pos = fruit.data.root_link_pos_w
  fruit_vel = fruit.data.root_link_lin_vel_w

  rel_pos = fruit_pos - ee_pos  # EE -> fruit center
  dist = rel_pos.norm(dim=-1)

  # Alignment of the blade with the EE -> fruit-center line.
  # (note: this might be too much since it might already be accounted for
  # in slicing.check_slice)
  axis = torch.tensor(blade_axis, dtype=ee_quat.dtype, device=env.device)
  axis = axis / axis.norm()
  blade_dir = quat_apply(ee_quat, axis.expand_as(rel_pos))
  cos_align = (blade_dir * rel_pos).sum(-1) / dist.clamp_min(1e-6)
  cos_align = torch.where(dist <= epsilon, torch.ones_like(cos_align), cos_align)
  align = cos_align.clamp(0.0, 1.0)

  # Cut check over the step, in the world frame (same as the termination),
  # so the speed criterion is on the blade alone.
  cut, miss_dist, blade_speed = slicing.check_slice(
    ee_pos - ee_vel * env.step_dt,
    ee_pos,
    fruit_pos - fruit_vel * env.step_dt,
    fruit_pos,
    dt=env.step_dt,
    hit_radius=epsilon,
    min_blade_speed=min_blade_speed,
  )

  # Interception: how close the EE came to the fruit center during the step.
  miss_excess = (miss_dist - epsilon).clamp_min(0.0)
  intercept = torch.exp(-miss_excess.square() / miss_std**2)

  # The blade must be moving fast enough for a cut rather than a touch.
  speed = (blade_speed / min_blade_speed).clamp(0.0, 1.0)

  dense = align * intercept * speed
  success = cut & (cos_align >= min_alignment)
  return dense + success.float()




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
