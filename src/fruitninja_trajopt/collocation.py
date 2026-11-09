"""Fixed-time Hermite-Simpson direct collocation for one intercept (CasADi + IPOPT).

Problem (arrival time T fixed, N intervals of h = T / N, h = control period):

  decision variables   q_k, v_k, a_k, u_k   k = 0..N   (position, velocity,
                                                          acceleration, PD target)
                       a_c,k                k = 0..N-1 (midpoint acceleration)

  minimize   sum_k h * ( w_tau |tau_k / tau_max|^2 + w_du |du_k/dt / vel_max|^2 )

  s.t.  q_0 = q(0), v_0 = v(0)
        inverse dynamics at knots and midpoints:
            RNEA(q, v, a) + armature * a = Kp (u - q) - Kd v
        Hermite-Simpson defects for q and v (v-dot = a)
        joint position / velocity / torque / PD-target limits
        cut at time T  (a conservative version of slicing.check_slice):
            |p_blade(q_N) - p_fruit(T)| <= hit_eps
            |v_blade(q_N, v_N)|        >= min_speed
            blade_axis(q_N) . z         >= min_blade_up

The decision variable ``u`` is the joint-position target of the simulator's PD
actuators, so a plan is executed by sending (a time-average of) ``u`` as actions.
Accelerations are decision variables, which turns the dynamics into the
inverse-dynamics form (no mass-matrix inverse inside the NLP).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from fruitninja_common import slicing
from fruitninja_trajopt.model import GRAVITY, NJ, ActuatorModel, IiwaModel


@dataclass
class TrajOptConfig:
  dt: float = 0.02  # control period; knot spacing (matches the 50 Hz env)
  # Margins: the sim checks a *step-averaged* blade speed and the PD loop does not
  # track perfectly, so aim inside the success region of slicing.check_slice.
  hit_eps: float = 0.5 * slicing.HIT_RADIUS
  min_speed: float = 1.5 * slicing.MIN_BLADE_SPEED
  min_blade_up: float = 0.8  # cos(max tilt); the RL success bonus uses 0.7
  vel_scale: float = 0.95  # fraction of the hardware joint velocity limits
  w_tau: float = 1.0
  w_du: float = 1.0
  max_iter: int = 300
  tol: float = 1e-4
  print_level: int = 0


@dataclass
class Guess:
  q: np.ndarray  # (N+1, NJ)
  v: np.ndarray
  a: np.ndarray
  u: np.ndarray
  a_mid: np.ndarray  # (N, NJ)


@dataclass
class Solution:
  success: bool  # solver converged AND the numeric post-check passed
  status: str
  n_steps: int
  T: float
  t: np.ndarray  # (N+1,)
  q: np.ndarray  # (N+1, NJ)
  v: np.ndarray
  u: np.ndarray
  tau: np.ndarray
  cost: float
  solve_time: float
  check: dict = field(default_factory=dict)

  def actions_abs(self) -> np.ndarray:
    """Joint-position targets for each of the N control steps (N, NJ).

    The sim holds a target for a whole step (zero-order hold) while the plan
    ramps ``u`` linearly, so the step's average, u_c = (u_k + u_{k+1}) / 2, is
    the matching constant.
    """
    return 0.5 * (self.u[:-1] + self.u[1:])


# --- formulas shared by the NLP (CasADi) and the numeric tests (numpy) ----------
def fruit_position(p0, v0, t):
  """Ballistic fruit position; the fruit only feels gravity."""
  p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
  return p0 + v0 * t + 0.5 * np.array([0.0, 0.0, -GRAVITY]) * t * t


def hermite_midpoint(xk, xn, fk, fn, h):
  """Hermite interpolant of the state at the interval midpoint."""
  return 0.5 * (xk + xn) + (h / 8.0) * (fk - fn)


def simpson_defect(xk, xn, fk, fc, fn, h):
  """Simpson quadrature of x_dot over one interval; zero for a consistent plan."""
  return xn - xk - (h / 6.0) * (fk + 4.0 * fc + fn)


def pd_torque(kp, kd, q, qd, u):
  return kp * (u - q) - kd * qd


def make_guess(
  model: IiwaModel, act: ActuatorModel, q0, v0, q_goal, n_steps: int, cfg: TrajOptConfig
) -> Guess:
  """Cubic Hermite move q0 -> q_goal that arrives with the blade moving.

  ``check_slice`` needs a blade speed of its own (world frame), so a seed that
  just parks the blade under the fruit would start far from feasible. The end
  velocity moves the blade point at 1.3 x ``min_speed`` in the direction that is
  cheapest in joint-velocity-limit units (top singular direction of J W,
  W = diag(velocity limits)); straight up is badly conditioned for this arm.
  """
  N, T = n_steps, n_steps * cfg.dt
  q0, v0, q_goal = (np.asarray(x, float) for x in (q0, v0, q_goal))
  W = np.diag(act.vel_limit)
  _, S, Vt = np.linalg.svd(model.ee_jacobian_num(q_goal) @ W)
  v_end = W @ Vt[0] * (1.3 * cfg.min_speed / S[0])
  if (model.ee_jacobian_num(q_goal) @ v_end)[2] < 0:  # prefer moving up over down
    v_end = -v_end
  s = np.linspace(0.0, 1.0, N + 1)[:, None]
  q = (
    (2 * s**3 - 3 * s**2 + 1) * q0
    + (s**3 - 2 * s**2 + s) * T * v0
    + (-2 * s**3 + 3 * s**2) * q_goal
    + (s**3 - s**2) * T * v_end
  )
  v = (
    (6 * s**2 - 6 * s) * q0
    + (3 * s**2 - 4 * s + 1) * T * v0
    + (-6 * s**2 + 6 * s) * q_goal
    + (3 * s**2 - 2 * s) * T * v_end
  ) / T
  a = (
    (12 * s - 6) * q0
    + (6 * s - 4) * T * v0
    + (-12 * s + 6) * q_goal
    + (6 * s - 2) * T * v_end
  ) / T**2
  # PD target that makes the guess dynamically consistent (feed-forward).
  u = np.empty_like(q)
  for k in range(N + 1):
    tau = model.inverse_dynamics(q[k], v[k], a[k]) + act.armature * a[k]
    u[k] = q[k] + (tau + act.kd * v[k]) / act.kp
  # Midpoint accelerations of the cubic (acceleration is linear in s).
  s_mid = (np.arange(N) + 0.5)[:, None] / N
  a_mid = (
    (12 * s_mid - 6) * q0
    + (6 * s_mid - 4) * T * v0
    + (-12 * s_mid + 6) * q_goal
    + (6 * s_mid - 2) * T * v_end
  ) / T**2
  return Guess(q, v, a, np.clip(u, act.q_lo, act.q_hi), a_mid)


def solve_fixed_time(
  model: IiwaModel,
  act: ActuatorModel,
  cfg: TrajOptConfig,
  q0,
  v0,
  fruit_p0,
  fruit_v0,
  n_steps: int,
  guess: Guess,
  fns: dict | None = None,
) -> Solution:
  import casadi as ca

  N = n_steps
  T = N * cfg.dt
  h = cfg.dt
  p_fruit_T = fruit_position(fruit_p0, fruit_v0, T)
  fns = fns or model.casadi_functions()
  id_fn = fns["inverse_dynamics"]

  col = lambda x: ca.DM(np.asarray(x, float).reshape(-1, 1))  # noqa: E731
  kp, kd, arm = col(act.kp), col(act.kd), col(act.armature)
  tau_max, v_max = col(act.effort), col(cfg.vel_scale * act.vel_limit)
  q_lo, q_hi = col(act.q_lo), col(act.q_hi)

  opti = ca.Opti()
  Q = opti.variable(NJ, N + 1)
  V = opti.variable(NJ, N + 1)
  A = opti.variable(NJ, N + 1)
  U = opti.variable(NJ, N + 1)
  Ac = opti.variable(NJ, N)

  def dyn_residual(q, v, a, u):
    return id_fn(q, v, a) + arm * a - pd_torque(kp, kd, q, v, u)

  cost = 0
  for k in range(N + 1):
    q, v, a, u = Q[:, k], V[:, k], A[:, k], U[:, k]
    tau = pd_torque(kp, kd, q, v, u)
    opti.subject_to(dyn_residual(q, v, a, u) == 0)
    opti.subject_to(opti.bounded(q_lo, q, q_hi))
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(q_lo, u, q_hi))
    opti.subject_to(opti.bounded(-tau_max, tau, tau_max))

  for k in range(N):
    qk, qn = Q[:, k], Q[:, k + 1]
    vk, vn = V[:, k], V[:, k + 1]
    ak, an = A[:, k], A[:, k + 1]
    uk, un = U[:, k], U[:, k + 1]
    ac = Ac[:, k]
    qc = hermite_midpoint(qk, qn, vk, vn, h)
    vc = hermite_midpoint(vk, vn, ak, an, h)
    uc = 0.5 * (uk + un)
    # x = [q; v], x_dot = [v; a]
    opti.subject_to(simpson_defect(qk, qn, vk, vc, vn, h) == 0)
    opti.subject_to(simpson_defect(vk, vn, ak, ac, an, h) == 0)
    opti.subject_to(dyn_residual(qc, vc, ac, uc) == 0)
    opti.subject_to(opti.bounded(-tau_max, pd_torque(kp, kd, qc, vc, uc), tau_max))
    # Smoothness cost (trapezoid): torque effort + rate of change of the target.
    tau_k = pd_torque(kp, kd, qk, vk, uk)
    tau_n = pd_torque(kp, kd, qn, vn, un)
    du = (un - uk) / h
    cost += 0.5 * h * cfg.w_tau * (
      ca.sumsqr(tau_k / tau_max) + ca.sumsqr(tau_n / tau_max)
    )
    cost += h * cfg.w_du * ca.sumsqr(du / v_max)

  # Initial condition.
  opti.subject_to(Q[:, 0] == col(q0))
  opti.subject_to(V[:, 0] == col(v0))

  # The cut at time T.
  p_blade = fns["ee_pos"](Q[:, N])
  v_blade = fns["ee_vel"](Q[:, N], V[:, N])
  axis = fns["blade_axis"](Q[:, N])
  opti.subject_to(ca.sumsqr(p_blade - col(p_fruit_T)) <= cfg.hit_eps**2)
  opti.subject_to(ca.sumsqr(v_blade) >= cfg.min_speed**2)
  opti.subject_to(axis[2] >= cfg.min_blade_up)

  opti.minimize(cost)
  for var, val in ((Q, guess.q), (V, guess.v), (A, guess.a), (U, guess.u), (Ac, guess.a_mid)):
    opti.set_initial(var, val.T)

  opti.solver(
    "ipopt",
    {"expand": True, "print_time": False},
    {
      "max_iter": cfg.max_iter,
      "tol": cfg.tol,
      "acceptable_tol": 10 * cfg.tol,
      "print_level": cfg.print_level,
      "sb": "yes",
    },
  )

  t0 = time.perf_counter()
  try:
    sol = opti.solve()
    value, stats = sol.value, sol.stats()
  except RuntimeError:  # infeasible / max iterations: keep the last iterate
    value, stats = opti.debug.value, opti.stats()
  solve_time = time.perf_counter() - t0
  status = str(stats.get("return_status", "unknown"))
  converged = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

  q = np.asarray(value(Q)).reshape(NJ, N + 1).T
  v = np.asarray(value(V)).reshape(NJ, N + 1).T
  u = np.asarray(value(U)).reshape(NJ, N + 1).T
  tau = act.pd_torque(q, v, u)
  plan = Solution(
    success=False,
    status=status,
    n_steps=N,
    T=T,
    t=np.arange(N + 1) * cfg.dt,
    q=q,
    v=v,
    u=u,
    tau=tau,
    cost=float(value(cost)),
    solve_time=solve_time,
  )
  plan.check = check_plan(model, act, cfg, plan, fruit_p0, fruit_v0)
  plan.success = converged and plan.check["ok"]
  return plan


def check_plan(
  model: IiwaModel,
  act: ActuatorModel,
  cfg: TrajOptConfig,
  plan: Solution,
  fruit_p0,
  fruit_v0,
  tol: float = 1e-3,
) -> dict:
  """Independent numeric check of a plan (does not trust the solver's verdict).

  ``would_slice`` applies the *shared* ``slicing.check_slice`` (true
  HIT_RADIUS / MIN_BLADE_SPEED) to the last control step of the plan, i.e.
  what the simulator's ``fruit_sliced`` termination will evaluate.
  """
  N, dt = plan.n_steps, cfg.dt
  p_b = np.array([model.ee_pos_num(q) for q in plan.q[-2:]])
  p_f = np.array([fruit_position(fruit_p0, fruit_v0, t) for t in plan.t[-2:]])
  would_slice, closest, speed = slicing.check_slice(p_b[0], p_b[1], p_f[0], p_f[1], dt)
  axis_z = float(model.blade_axis_num(plan.q[-1])[2])
  out = {
    "would_slice": bool(would_slice),
    "closest_distance": float(closest),
    "blade_speed": float(speed),
    "blade_up": axis_z,
    "end_distance": float(np.linalg.norm(p_b[1] - p_f[1])),
    "max_vel_ratio": float((np.abs(plan.v) / act.vel_limit).max()),
    "max_torque_ratio": float((np.abs(plan.tau) / act.effort).max()),
    "min_limit_margin": float(
      min((plan.q - act.q_lo).min(), (act.q_hi - plan.q).min())
    ),
  }
  out["ok"] = bool(
    out["would_slice"]
    and out["blade_up"] >= 0.7 - tol  # the RL success bonus's min_alignment
    and out["max_vel_ratio"] <= 1.0 + tol
    and out["max_torque_ratio"] <= 1.0 + tol
    and out["min_limit_margin"] >= -tol
  )
  return out
