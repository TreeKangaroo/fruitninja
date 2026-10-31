"""The cut condition: did the blade slice the fruit during a time interval?

This is the single definition of success used by the RL reward/termination,
the evaluation metric, and (later) trajectory optimization, so that both
methods are judged identically.

A slice happens when, during one interval [t0, t1]:
  1. the blade point passes within ``hit_radius`` of the fruit center, and
  2. the blade point is moving at least ``min_blade_speed``.

Checking only the endpoints t0 and t1 would miss fast pass-throughs (at 50 Hz a
falling fruit moves several cm per step), so condition 1 uses the closest
approach of the two points, assuming both move linearly within the interval.

The functions only use arithmetic, ``.sum``, and ``.clip``, so they work on
both torch tensors and numpy arrays with shape (..., 3).
"""

from fruitninja_common.fruit import FRUIT_RADIUS

# Blade must pass within this distance of the fruit center. Using the fruit
# radius means "the blade point goes through the fruit". Tighten this to
# require cutting closer to the midpoint (proposal Sec. 4.1).
HIT_RADIUS = FRUIT_RADIUS  # [m]
# Minimum blade speed for a cut rather than a touch.
MIN_BLADE_SPEED = 0.5  # [m/s]


def _norm(v):
  return (v * v).sum(-1) ** 0.5


def closest_approach(blade_t0, blade_t1, fruit_t0, fruit_t1):
  """Minimum blade-to-fruit-center distance over the interval [t0, t1]."""
  rel_t0 = blade_t0 - fruit_t0
  rel_t1 = blade_t1 - fruit_t1
  d = rel_t1 - rel_t0
  # Minimize |rel_t0 + s*d| over s in [0, 1].
  s = (-(rel_t0 * d).sum(-1) / ((d * d).sum(-1) + 1e-12)).clip(0.0, 1.0)
  return _norm(rel_t0 + s[..., None] * d)


def check_slice(
  blade_t0,
  blade_t1,
  fruit_t0,
  fruit_t1,
  dt: float,
  hit_radius: float = HIT_RADIUS,
  min_blade_speed: float = MIN_BLADE_SPEED,
):
  """Returns (sliced, closest_distance, blade_speed) for the interval."""
  distance = closest_approach(blade_t0, blade_t1, fruit_t0, fruit_t1)
  blade_speed = _norm(blade_t1 - blade_t0) / dt
  sliced = (distance <= hit_radius) & (blade_speed >= min_blade_speed)
  return sliced, distance, blade_speed
