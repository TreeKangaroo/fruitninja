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
  throws, widen the velocity ranges and add the result to ``TRAJECTORIES``.

  ``weight`` is how often this range is picked, relative to the others in
  ``TRAJECTORIES`` (0 = never).
  """

  x: tuple[float, float] = (0.45, 0.70)  # [m]
  y: tuple[float, float] = (-0.30, 0.30)  # [m]
  z: tuple[float, float] = (1.40, 1.70)  # [m]
  vx: tuple[float, float] = (0.0, 0.0)  # [m/s]
  vy: tuple[float, float] = (0.0, 0.0)  # [m/s]
  vz: tuple[float, float] = (0.0, 0.0)  # [m/s]
  weight: float = 1.0


# The "xvel" trajectory: the same drop, plus a constant velocity along x towards
# the robot. Release height, y and fall time are those of FruitDropRange(), so
# only the shape of the path changes. The release x is moved out to compensate
# for the drift: every fruit crosses blade height (z = 0.70 m, 0.38-0.45 s after
# release) at x = 0.45-0.70 m, as in the straight drop. If you change vx, set
# x = (0.45 - vx_min * 0.452, 0.70 - vx_max * 0.378) to stay in that band
# (for vx <= 0).
X_VELOCITY_DROP = FruitDropRange(x=(0.79, 0.885), vx=(-0.75, -0.50))
# (0.79, 0.885)

# The "zvel" trajectory: the same drop, but tossed up (vz > 0) or thrown down
# (vz < 0). x, y and vx are those of FruitDropRange(), so the fruit still falls
# straight through x = 0.45-0.70 m; only the timing changes. The release z is
# moved down to compensate for the toss: the highest point of every path
# (z0 + vz^2 / 2g) is still 1.40-1.70 m, so the fruit reaches blade height at the
# straight drop's speed (3.7-4.4 m/s), but 0.26-0.60 s after release instead of
# 0.38-0.45 s. Thrown-down fruit leave the robot less time, tossed-up fruit more.
# Air time grows to up to 0.73 s, so more fruit share the pool's slots. If you
# change vz, set z = (1.40 - min|vz|^2 / 2g, 1.70 - max|vz|^2 / 2g)
# (min|vz| = 0 when the range spans 0); it needs max|vz| <~ 2.4 m/s.
Z_VELOCITY_DROP = FruitDropRange(z=(1.40, 1.585), vz=(-1.5, 1.5))

# The "yvel" trajectories: the same drop, plus a constant sideways velocity, like
# xvel but along y. Release height, x and fall time are those of
# FruitDropRange(). The release y is moved to one side to compensate for the
# drift: every fruit crosses blade height (0.38-0.45 s after release) at
# y = -0.30-0.30 m, as in the straight drop. A single range with both signs of vy
# would leave only y = +-0.07 m to release from, so there is one entry per
# direction, mirror images of each other. If you change vy (for vy >= 0), set
# y = (-0.30 - vy_min * 0.378, 0.30 - vy_max * 0.452) and mirror it for the other.
Y_VELOCITY_POS_DROP = FruitDropRange(y=(-0.489, -0.039), vy=(0.50, 0.75), weight=0.5)
Y_VELOCITY_NEG_DROP = FruitDropRange(y=(0.039, 0.489), vy=(-0.75, -0.50), weight=0.5)

# What the robot trains on: every fruit independently picks one of these when it
# spawns, so a single episode (and the air at one moment) mixes them. The two
# yvel directions have half weight each, so sideways fruit as a whole are as
# common as each other type.
TRAJECTORIES: dict[str, FruitDropRange] = {
  "straight": FruitDropRange(),
  "xvel": X_VELOCITY_DROP,
  "zvel": Z_VELOCITY_DROP,
  "yvel_pos": Y_VELOCITY_POS_DROP,
  "yvel_neg": Y_VELOCITY_NEG_DROP,
}


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
  # places each fruit according to one of TRAJECTORIES when it spawns.
  return EntityCfg(spec_fn=partial(get_fruit_spec, rgba=rgba))
