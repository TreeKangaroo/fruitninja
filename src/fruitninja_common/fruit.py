"""The fruit: a free-falling sphere, plus the region it is dropped from.

Shared by RL and trajectory optimization. All positions are in the robot base
frame (the robot sits at the origin of each env, unrotated).
"""

from dataclasses import dataclass

import mujoco

from mjlab.entity import EntityCfg

FRUIT_RADIUS = 0.04  # [m] roughly an apple
FRUIT_MASS = 0.15  # [kg]


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
  rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.1, 1.0),
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


def get_fruit_cfg() -> EntityCfg:
  # The initial position is irrelevant: a reset event drops the fruit at a
  # random point of FruitDropRange at the start of every episode.
  return EntityCfg(spec_fn=get_fruit_spec)
