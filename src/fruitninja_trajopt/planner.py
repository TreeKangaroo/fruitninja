"""Outer search over arrival times (proposal Sec. 3.2.1).

For a fruit with known state (p0, v0) and the arm at (q0, qd0):

  1. candidate arrival times T = n * dt, n = n_min..n_max; the fruit must still
     be above ``z_floor`` at T;
  2. for each, IK for "blade point at the fruit position at T, blade up" gives a
     reachability filter and the seed for the collocation;
  3. a cheap velocity-limit prefilter drops arrival times that would need a
     joint to move faster than its limit on average (heuristic: uses the IK
     solution closest to q0);
  4. one fixed-time collocation per remaining candidate (at most ``max_solves``,
     spread over the window); the best verified plan wins.

Because T is a multiple of the control period and knots sit on control steps,
the plan can be replayed step by step in the 50 Hz environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from fruitninja_trajopt.collocation import (
  Solution,
  TrajOptConfig,
  fruit_position,
  make_guess,
  solve_fixed_time,
)
from fruitninja_trajopt.ik import solve_ik
from fruitninja_trajopt.model import GRAVITY, ActuatorModel, IiwaModel


@dataclass
class PlannerConfig:
  traj: TrajOptConfig = field(default_factory=TrajOptConfig)
  min_steps: int = 8  # earliest arrival: 8 * 20 ms
  z_floor: float = 0.15 + 0.03  # fruit must be above this at T (MISS_HEIGHT + margin)
  stride: int = 1  # consider every ``stride``-th arrival step
  max_solves: int = 6  # cap on NLP solves per plan
  prune_by_velocity: bool = True
  # Blade points farther than this from the shoulder are skipped without IK.
  # (iiwa14 reach is 0.82 m to the flange; the blade-up pose reaches a bit less.)
  max_reach: float = 0.9
  select: str = "min_cost"  # "min_cost" or "earliest"


@dataclass
class PlanResult:
  best: Solution | None
  candidates: list[Solution]
  n_considered: int  # arrival times in the window
  n_ik_ok: int  # ... that IK can reach with the blade up
  n_velocity_ok: int  # ... that also pass the velocity prefilter
  plan_time: float  # total wall-clock time [s]
  ik_time: float
  nlp_time: float

  @property
  def success(self) -> bool:
    return self.best is not None


def flight_time_to_height(p0, v0, z: float) -> float:
  """Time at which the ballistic fruit descends through height ``z``."""
  z0, vz = float(p0[2]), float(v0[2])
  disc = vz * vz + 2.0 * GRAVITY * (z0 - z)
  return (vz + np.sqrt(max(disc, 0.0))) / GRAVITY if z0 > z else 0.0


@dataclass
class Screening:
  steps: list[int]  # arrival steps in the window
  ik_ok: list[tuple[int, np.ndarray]]  # reachable with the blade up: (step, q_goal)
  vel_ok: list[tuple[int, np.ndarray]]  # ... and passing the velocity prefilter
  time: float


def screen_arrival_times(
  model: IiwaModel, act: ActuatorModel, q0, fruit_p0, fruit_v0, cfg: PlannerConfig
) -> Screening:
  """Steps 1-3 of the planner: no NLP, no CasADi."""
  tc = cfg.traj
  t0 = time.perf_counter()
  n_max = int(flight_time_to_height(fruit_p0, fruit_v0, cfg.z_floor) / tc.dt)
  steps = list(range(cfg.min_steps, n_max + 1, cfg.stride))
  ik_ok, vel_ok = [], []
  for n in steps:
    T = n * tc.dt
    p = fruit_position(fruit_p0, fruit_v0, T)
    if np.linalg.norm(p - model.shoulder) > cfg.max_reach:
      continue
    ik = solve_ik(model, act, p, q0, min_blade_up=tc.min_blade_up)
    if not ik.success:
      continue
    ik_ok.append((n, ik.q))
    avg_speed = np.abs(ik.q - q0) / T  # a joint can't average more than its limit
    if cfg.prune_by_velocity and np.any(avg_speed > tc.vel_scale * act.vel_limit):
      continue
    vel_ok.append((n, ik.q))
  return Screening(steps, ik_ok, vel_ok, time.perf_counter() - t0)


def plan_intercept(
  model: IiwaModel,
  act: ActuatorModel,
  q0,
  v0,
  fruit_p0,
  fruit_v0,
  cfg: PlannerConfig | None = None,
  fns: dict | None = None,
) -> PlanResult:
  cfg = cfg or PlannerConfig()
  tc = cfg.traj
  q0, v0 = np.asarray(q0, float), np.asarray(v0, float)
  t_start = time.perf_counter()

  scr = screen_arrival_times(model, act, q0, fruit_p0, fruit_v0, cfg)

  # Collocation for (a spread of) the remaining candidates.
  chosen = scr.vel_ok
  if len(chosen) > cfg.max_solves:
    idx = np.unique(np.linspace(0, len(chosen) - 1, cfg.max_solves).round().astype(int))
    chosen = [chosen[i] for i in idx]

  t_nlp0 = time.perf_counter()
  candidates: list[Solution] = []
  if chosen:
    fns = fns or model.casadi_functions()
  for n, q_goal in chosen:
    guess = make_guess(model, act, q0, v0, q_goal, n, tc)
    candidates.append(
      solve_fixed_time(model, act, tc, q0, v0, fruit_p0, fruit_v0, n, guess, fns)
    )
  nlp_time = time.perf_counter() - t_nlp0

  good = [c for c in candidates if c.success]
  best = None
  if good:
    best = min(good, key=(lambda c: c.cost) if cfg.select == "min_cost" else (lambda c: c.T))
  return PlanResult(
    best=best,
    candidates=candidates,
    n_considered=len(scr.steps),
    n_ik_ok=len(scr.ik_ok),
    n_velocity_ok=len(scr.vel_ok),
    plan_time=time.perf_counter() - t_start,
    ik_time=scr.time,
    nlp_time=nlp_time,
  )
