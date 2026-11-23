"""A pool of falling fruit, spawned over the course of one episode.

One episode is a stream of fruit. Each env owns ``NUM_FRUIT_SLOTS`` fruit
bodies (``fruit_0`` ...) that are reused: a spawn takes a free slot, drops it
from a random point, and the slot becomes free again once that fruit hits the
floor. Fruit do not collide with anything; slicing is purely geometric
(``slicing.check_slice``), so a sliced fruit keeps falling undisturbed.

Per fruit: unsliced (red) -> sliced (green, still falling) -> landed (hidden,
slot free). Every slot has two bodies, a red ``fruit_k`` and a green
``sliced_fruit_k``. Only one of them is in play at a time; the other is parked
under the floor, out of sight. Cutting a fruit hands its position and velocity
to the green twin, so the fall continues unchanged. (Recoloring one body with
per-env ``geom_rgba`` would work for the offscreen renderer, but the viser
viewer rebuilds every mesh whenever a color changes.)

The pool is a class-based *reset* event (``mdp.FruitPool``, named
``"fruit_pool"`` in the env cfg): calling it clears the env's state and spawns
the first fruit. Later spawns come from the interval event ``spawn_fruit``.
Rewards, observations and metrics find the instance with ``get_fruit_pool``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fruitninja_common import slicing
from fruitninja_common.fruit import FRUIT_RADIUS, FruitDropRange

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

__all__ = ["FruitPool", "get_fruit_pool", "spawn_fruit"]

POOL_TERM_NAME = "fruit_pool"
PARK_HEIGHT = -5.0  # [m] below the floor, relative to the env origin


class FruitPool:
  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.names: tuple[str, ...] = tuple(p["fruit_names"])  # red bodies
    self.sliced_names: tuple[str, ...] = tuple(p["sliced_fruit_names"])  # green twins
    self.drop: FruitDropRange = p["drop"]
    self.ee_cfg = p["asset_cfg"]
    self.hit_radius: float = p.get("hit_radius", slicing.HIT_RADIUS)
    self.min_blade_speed: float = p.get("min_blade_speed", slicing.MIN_BLADE_SPEED)
    n, k, dev = env.num_envs, len(self.names), env.device

    # Per-fruit state, (envs, slots).
    self.active = torch.zeros(n, k, dtype=torch.bool, device=dev)  # in the air
    self.sliced = torch.zeros(n, k, dtype=torch.bool, device=dev)
    self.prev_fruit_pos = torch.zeros(n, k, 3, device=dev)
    # Per-episode counters, (envs,).
    self.spawned_count = torch.zeros(n, dtype=torch.long, device=dev)
    self.sliced_count = torch.zeros(n, dtype=torch.long, device=dev)
    self.missed_count = torch.zeros(n, dtype=torch.long, device=dev)
    # Blade state at the previous step. False right after a reset (no previous
    # step to compare to, so no cut can be detected on the first step).
    self.prev_blade_pos = torch.zeros(n, 3, device=dev)
    self.has_prev = torch.zeros(n, dtype=torch.bool, device=dev)
    # Outputs of the latest update().
    self.new_slices = torch.zeros(n, device=dev)  # fruit sliced this step
    self.new_misses = torch.zeros(n, device=dev)  # unsliced fruit that landed this step
    self.blade_speed = torch.zeros(n, device=dev)
    # Closest blade-fruit approach over the latest step, (envs, slots). Only
    # meaningful for fruit in play (and not on the first step after a reset).
    self.closest_distance = torch.zeros(n, k, device=dev)
    # For reward shaping: the fruit that were in the air and unsliced at the
    # start of the step (mask), with their positions/velocities at the end of it.
    self.shaping_mask = torch.zeros(n, k, device=dev)  # float 0/1
    self.shaping_pos = torch.zeros(n, k, 3, device=dev)
    self.shaping_vel = torch.zeros(n, k, 3, device=dev)

    self._updated_at = -1

  # Reset event: wipe this env's fruit and drop the first one.

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **_) -> None:
    self.active[env_ids] = False
    self.sliced[env_ids] = False
    self.spawned_count[env_ids] = 0
    self.sliced_count[env_ids] = 0
    self.missed_count[env_ids] = 0
    self.has_prev[env_ids] = False
    for k in range(len(self.names)):
      self._park(env, k, env_ids)
    self.spawn(env, env_ids)

  # State access.

  def positions(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """(envs, slots, 3) world-frame centers of the bodies in play (garbage for
    free slots: mask with ``active``)."""
    return self._live(env, "root_link_pos_w")

  def velocities(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return self._live(env, "root_link_lin_vel_w")

  def _live(self, env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
    whole = torch.stack([getattr(env.scene[n].data, field) for n in self.names], 1)
    cut = torch.stack([getattr(env.scene[n].data, field) for n in self.sliced_names], 1)
    return torch.where(self.sliced[..., None], cut, whole)

  # Spawning.

  def spawn(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Drop one new fruit in each of ``env_ids``, using each env's first free slot.

    Envs with every slot busy (already ``len(names)`` fruit in the air) skip
    this spawn, which caps the number of simultaneous fruit.
    """
    free = ~self.active[env_ids]
    has_free = free.any(1)
    env_ids = env_ids[has_free]
    free = free[has_free]
    n = len(env_ids)
    if n == 0:
      return
    slot = free.to(torch.int8).argmax(1)  # first free slot

    def uniform(lo_hi: tuple[float, float]) -> torch.Tensor:
      return torch.empty(n, device=env.device).uniform_(*lo_hi)

    d = self.drop
    pos = torch.stack([uniform(d.x), uniform(d.y), uniform(d.z)], 1)
    pos = pos + env.scene.env_origins[env_ids]
    state = torch.zeros(n, 13, device=env.device)
    state[:, 0:3] = pos
    state[:, 3] = 1.0  # identity quaternion (w, x, y, z)
    state[:, 7:10] = torch.stack([uniform(d.vx), uniform(d.vy), uniform(d.vz)], 1)

    for k, name in enumerate(self.names):
      sel = (slot == k).nonzero().squeeze(-1)
      if sel.numel() == 0:
        continue
      ids = env_ids[sel]
      env.scene[name].write_root_state_to_sim(state[sel], ids)
      self.active[ids, k] = True
      self.sliced[ids, k] = False
      self.prev_fruit_pos[ids, k] = pos[sel]
    self.spawned_count[env_ids] += 1

  # Per-step bookkeeping: cut detection, landing, swapping cut fruit to their
  # green twins. Idempotent within a
  # step, so every reward/metric can call it and the first caller does the work.

  def update(self, env: ManagerBasedRlEnv) -> None:
    if self._updated_at == env.common_step_counter:
      return
    self._updated_at = env.common_step_counter

    pos = self.positions(env)
    vel = self.velocities(env)
    robot = env.scene[self.ee_cfg.name]
    blade = robot.data.site_pos_w[:, self.ee_cfg.site_ids].squeeze(1)

    # Shaping targets: every fruit still in the air and unsliced.
    candidate = self.active & ~self.sliced
    self.shaping_mask = candidate.float()
    self.shaping_pos = pos
    self.shaping_vel = vel

    # Cuts, for all slots at once. Blade is (envs, 1, 3), fruit (envs, slots, 3).
    cut, self.closest_distance, speed = slicing.check_slice(
      self.prev_blade_pos[:, None],
      blade[:, None],
      self.prev_fruit_pos,
      pos,
      dt=env.step_dt,
      hit_radius=self.hit_radius,
      min_blade_speed=self.min_blade_speed,
    )
    self.blade_speed = torch.where(self.has_prev, speed[:, 0], torch.zeros_like(speed[:, 0]))
    hit = cut & candidate & self.has_prev[:, None]
    self.new_slices = hit.sum(1).float()
    self.sliced_count += hit.sum(1)

    # Fruit that reached the floor (sliced or not) free their slot.
    floor = env.scene.env_origins[:, 2:3]
    landed = self.active & (pos[..., 2] - floor <= FRUIT_RADIUS)
    missed = landed & ~self.sliced & ~hit
    self.new_misses = missed.sum(1).float()
    self.missed_count += missed.sum(1)
    self.active &= ~landed

    self.prev_fruit_pos = pos.clone()
    self.prev_blade_pos = blade.clone()
    self.has_prev[:] = True

    # Hand freshly cut fruit to their green twins, then clear the landed ones.
    # (Both need the ids on the host, so only do the work when there is any.)
    if hit.any():
      for k in range(len(self.names)):
        ids = hit[:, k].nonzero().squeeze(-1)
        if ids.numel() > 0:
          self._swap_to_sliced(env, k, ids, pos[ids, k], vel[ids, k])
    self.sliced |= hit
    if landed.any():
      for k in range(len(self.names)):
        ids = landed[:, k].nonzero().squeeze(-1)
        if ids.numel() > 0:
          self._park(env, k, ids)

  # Internals.

  def _swap_to_sliced(self, env, k: int, ids, pos, vel) -> None:
    """Continue fruit ``k``'s fall in its green twin and park the red body."""
    state = torch.zeros(len(ids), 13, device=env.device)
    # Positions read after a physics step lag the true position by one physics
    # substep; carry that over so the fruit does not jump back.
    state[:, 0:3] = pos + vel * env.physics_dt
    state[:, 3] = 1.0
    state[:, 7:10] = vel
    env.scene[self.sliced_names[k]].write_root_state_to_sim(state, ids)
    self._park_one(env, self.names[k], ids)

  def _park(self, env, k: int, env_ids: torch.Tensor) -> None:
    """Hide both bodies of slot ``k`` in ``env_ids``: under the floor, at rest."""
    self._park_one(env, self.names[k], env_ids)
    self._park_one(env, self.sliced_names[k], env_ids)

  def _park_one(self, env, name: str, env_ids: torch.Tensor) -> None:
    state = torch.zeros(len(env_ids), 13, device=env.device)
    state[:, 0:3] = env.scene.env_origins[env_ids]
    state[:, 2] += PARK_HEIGHT
    state[:, 3] = 1.0
    env.scene[name].write_root_state_to_sim(state, env_ids)


def get_fruit_pool(env: ManagerBasedRlEnv, update: bool = False) -> FruitPool:
  """The env's FruitPool. With ``update=True``, first run this step's bookkeeping
  (use it from rewards/metrics; observations just read the current state)."""
  pool = env.event_manager.get_term_cfg(POOL_TERM_NAME).func
  if update:
    pool.update(env)
  return pool


def spawn_fruit(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Interval event: drop the next fruit in each env whose timer ran out."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  get_fruit_pool(env).spawn(env, env_ids)
