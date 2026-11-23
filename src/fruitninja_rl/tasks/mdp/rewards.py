"""Rewards. mjlab multiplies every term by ``weight * step_dt``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fruitninja_common import slicing
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from .fruit_pool import get_fruit_pool

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "slices_this_step",
  "fruits_missed_this_step",
  "ee_fruit_proximity",
  "slice_object_reward",
  "staged_slice_reward",
  "joint_vel_limits",
]


def slices_this_step(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of fruit sliced this step (usually 0 or 1)."""
  return get_fruit_pool(env, update=True).new_slices


def fruits_missed_this_step(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of unsliced fruit that reached the floor this step."""
  return get_fruit_pool(env, update=True).new_misses


# Dense shaping. Several fruit are in the air at once, so each term is computed
# for every fruit (envs, slots) and the best one counts (max over fruit). That
# way the policy is free to chase whichever fruit it can actually reach instead
# of being tied to e.g. the lowest one. Fruit already sliced earn nothing.


def _reaching(
  env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """(envs, slots): exp(-d^2 / std^2), d = EE-to-fruit distance; 0 for non-targets."""
  pool = get_fruit_pool(env, update=True)
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  dist_sq = (pool.shaping_pos - ee_pos[:, None]).square().sum(-1)
  return torch.exp(-dist_sq / std**2) * pool.shaping_mask


def _slicing(
  env: ManagerBasedRlEnv,
  miss_std: float,
  asset_cfg: SceneEntityCfg,
  epsilon: float,
  min_blade_speed: float,
  min_alignment: float,
  blade_axis: tuple[float, float, float],
) -> torch.Tensor:
  """(envs, slots): the slice-shaping score of each fruit, in [0, 2]; 0 for non-targets.

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

  Returns ``dense + bonus``: ``dense`` is the product of the three soft terms;
  ``bonus`` is 1.0 when ``slicing.check_slice`` reports a cut (d <= epsilon and
  blade speed >= min_blade_speed) and alignment >= ``min_alignment``. The start
  of the step is reconstructed from the current positions and velocities
  (p_prev = p - v * step_dt).
  """
  pool = get_fruit_pool(env, update=True)
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)[:, None]  # (B,1,3)
  ee_quat = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)[:, None]
  ee_vel = robot.data.site_lin_vel_w[:, asset_cfg.site_ids].squeeze(1)[:, None]
  fruit_pos = pool.shaping_pos  # (B,K,3)
  fruit_vel = pool.shaping_vel

  rel_pos = fruit_pos - ee_pos  # EE -> fruit center
  dist = rel_pos.norm(dim=-1)

  # Alignment of the blade with the EE -> fruit-center line.
  # (note: this might be too much since it might already be accounted for
  # in slicing.check_slice)
  axis = torch.tensor(blade_axis, dtype=ee_quat.dtype, device=env.device)
  axis = axis / axis.norm()
  blade_dir = quat_apply(ee_quat.expand(-1, rel_pos.shape[1], -1), axis.expand_as(rel_pos))
  cos_align = (blade_dir * rel_pos).sum(-1) / dist.clamp_min(1e-6)
  cos_align = torch.where(dist <= epsilon, torch.ones_like(cos_align), cos_align)
  align = cos_align.clamp(0.0, 1.0)

  # Cut check over the step, in the world frame (same as the fruit pool), so
  # the speed criterion is on the blade alone.
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
  return (dense + success.float()) * pool.shaping_mask


def ee_fruit_proximity(
  env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Dense shaping: exp(-d^2 / std^2) for the closest-to-the-blade unsliced fruit."""
  return _reaching(env, std, asset_cfg).max(dim=1).values


def slice_object_reward(
  env: ManagerBasedRlEnv,
  miss_std: float,
  asset_cfg: SceneEntityCfg,
  epsilon: float = slicing.HIT_RADIUS,
  min_blade_speed: float = slicing.MIN_BLADE_SPEED,
  min_alignment: float = 0.7,  # alignment along the blade axes
  blade_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
  """Shaped reward for slicing through a fruit during this step: the best
  fruit's score from ``_slicing`` (alignment x interception x speed, plus 1 on a cut)."""
  return _slicing(
    env, miss_std, asset_cfg, epsilon, min_blade_speed, min_alignment, blade_axis
  ).max(dim=1).values


def staged_slice_reward(
  env: ManagerBasedRlEnv,
  reaching_std: float,
  miss_std: float,
  asset_cfg: SceneEntityCfg,
  epsilon: float = slicing.HIT_RADIUS,
  min_blade_speed: float = slicing.MIN_BLADE_SPEED,
  min_alignment: float = 0.7,
  blade_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
  """Staged reward per fruit: reach it, then slice it. Best fruit counts.

  For each unsliced fruit in the air, ``reaching * (1 + slicing)``:

  * ``reaching`` = ``exp(-d^2 / reaching_std^2)``, in [0, 1]. Gives the
    approach signal far from the fruit.
  * ``slicing`` in [0, 2]: only pays when the blade is aligned, intercepting,
    and fast, plus 1 on an actual cut.

  Per fruit the result is in [0, 3]; the reward is the max over fruit. Getting
  close without cutting earns at most 1 per step, while a real slice earns up
  to 3, so the policy cannot do as well by just hovering next to a fruit.
  Multiplying by ``reaching`` also keeps the slicing term from paying out far
  from the fruit.
  """
  reaching = _reaching(env, reaching_std, asset_cfg)
  slicing_term = _slicing(
    env, miss_std, asset_cfg, epsilon, min_blade_speed, min_alignment, blade_axis
  )
  return (reaching * (1.0 + slicing_term)).max(dim=1).values


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
