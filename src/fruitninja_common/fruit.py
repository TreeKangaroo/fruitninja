"""The fruit: a free-falling sphere, plus the region it is dropped from.

Shared by RL and trajectory optimization. All positions are in the robot base
frame (the robot sits at the origin of each env, unrotated).
"""

from dataclasses import dataclass
from functools import partial

import mujoco

from mjlab.entity import EntityCfg

FRUIT_RADIUS = 0.04  # [m] roughly an apple
FRUIT_MASS = 0.15  # [kg]
# Height reference for planners: a fruit whose center is below this counts as
# lost. (The RL fruit pool is stricter about "landed": it frees a slot only at
# FRUIT_RADIUS above the floor, so planning against MISS_HEIGHT is conservative.)
MISS_HEIGHT = 0.15  # [m]

FRUIT_COLOR = (0.9, 0.2, 0.1, 1.0)  # whole fruit
SLICED_COLOR = (0.2, 0.85, 0.2, 1.0)  # after the blade has cut it

# RL: a pool of reusable fruit bodies. A fruit lives ~0.6 s (fall from ~1.5 m)
# and the pool size is the cap on fruit in the air at once: a spawn is skipped
# while every slot is busy.
NUM_FRUIT_SLOTS = 4
# Gap between consecutive drops, uniform. Mean 0.375 s with a ~0.6 s fall is
# about 1.6 fruit in the air on average, and 3-4 in bursts. Shorten it for a
# harder game (e.g. --env.events.spawn-fruit.interval-range-s 0.1 0.4).
SPAWN_INTERVAL_S = (0.15, 0.6)  # [s]


def fruit_name(slot: int) -> str:
  return f"fruit_{slot}"


def sliced_fruit_name(slot: int) -> str:
  """The green twin of ``fruit_name(slot)``: it takes over when the fruit is cut."""
  return f"sliced_fruit_{slot}"


@dataclass(frozen=True)
class FruitDropRange:
  """Uniform ranges for the fruit's initial position and velocity.

  The default is the proposal's base case: the fruit starts at rest and
  falls straight down, through the robot's reachable workspace. For 3D
  throws, widen the velocity ranges.
  """

  x: tuple[float, float] = (0.45, 0.70)  # [m]
  y: tuple[float, float] = (-0.30, 0.30)  # [m]
  z: tuple[float, float] = (1.40, 1.70)  # [m]
  vx: tuple[float, float] = (0.0, 0.0)  # [m/s]
  vy: tuple[float, float] = (0.0, 0.0)  # [m/s]
  vz: tuple[float, float] = (0.0, 0.0)  # [m/s]


def get_fruit_spec(
  radius: float = FRUIT_RADIUS,
  mass: float = FRUIT_MASS,
  rgba: tuple[float, float, float, float] = FRUIT_COLOR,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="fruit")
  body.add_freejoint(name="fruit_joint")
  body.add_geom(
    name="fruit_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0, 0),
    mass=mass,
    rgba=rgba,
    # No collisions: the fruit is a "ghost" that falls under gravity only.
    # Slicing is decided geometrically (see slicing.py), so the arm never
    # knocks the fruit away and its trajectory stays exactly ballistic.
    contype=0,
    conaffinity=0,
  )
  return spec


def get_fruit_cfg(rgba: tuple[float, float, float, float] = FRUIT_COLOR) -> EntityCfg:
  # The initial position is irrelevant: the fruit pool (tasks/mdp/fruit_pool.py)
  # places each fruit at a random point of FruitDropRange when it spawns.
  return EntityCfg(spec_fn=partial(get_fruit_spec, rgba=rgba))
