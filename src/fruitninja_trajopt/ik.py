"""Inverse kinematics used to seed the collocation (proposal Sec. 3.2.1).

Finds a joint configuration that puts the blade point on the intercept point
with the blade pointing up, close to the current configuration. Plain bounded
least squares; the arm is redundant, so a weak pull towards ``q_ref`` picks
among the 1-D family of solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from fruitninja_trajopt.model import ActuatorModel, IiwaModel


@dataclass
class IkResult:
  q: np.ndarray
  pos_error: float  # [m]
  blade_up: float  # cos(angle between blade axis and +z)
  success: bool


def solve_ik(
  model: IiwaModel,
  act: ActuatorModel,
  p_target: np.ndarray,
  q_ref: np.ndarray,
  min_blade_up: float = 0.8,
  pos_tol: float = 2e-3,
  w_ori: float = 0.3,
  w_reg: float = 0.02,
  restarts: int = 6,
  seed: int = 0,
) -> IkResult:
  p_target = np.asarray(p_target, dtype=float)
  lo, hi = act.q_lo, act.q_hi
  rng = np.random.default_rng(seed)

  def residual(q):
    p = model.ee_pos_num(q)
    up = model.blade_axis_num(q)[2]
    return np.concatenate([p - p_target, [w_ori * (1.0 - up)], w_reg * (q - q_ref)])

  seeds = [np.clip(q_ref, lo, hi)]
  for _ in range(restarts):
    seeds.append(np.clip(q_ref + rng.normal(0.0, 0.6, q_ref.shape), lo, hi))

  best: IkResult | None = None
  for q_init in seeds:
    sol = least_squares(residual, q_init, bounds=(lo, hi), max_nfev=60)
    err = float(np.linalg.norm(model.ee_pos_num(sol.x) - p_target))
    up = float(model.blade_axis_num(sol.x)[2])
    res = IkResult(sol.x, err, up, err <= pos_tol and up >= min_blade_up)
    # Prefer successful solutions, then the one closest to q_ref.
    key = (not res.success, np.linalg.norm(res.q - q_ref))
    if best is None or key < (not best.success, np.linalg.norm(best.q - q_ref)):
      best = res
    if res.success and np.linalg.norm(res.q - q_ref) < 0.5:
      break  # good enough; skip remaining restarts
  assert best is not None
  return best
