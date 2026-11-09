"""Offline feasibility / compute-cost study (no simulator needed).

  uv run python -m fruitninja_trajopt.feasibility --n 50 [--ready-pose current]

For random fruit drops from ``FruitDropRange``, runs the planner from the ready
pose and reports how many fruits have a verified intercept plan. That fraction
is the ceiling for *any* method on this arm / blade / cut condition: report RL's
success rate relative to it.
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

from fruitninja_common.fruit import FruitDropRange
from fruitninja_trajopt.model import (
  BLADE_UP_READY_POS,
  ActuatorModel,
  IiwaModel,
  default_joint_pos,
)
from fruitninja_trajopt.planner import PlannerConfig, plan_intercept


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=30)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--ready-pose", choices=("blade_up", "current"), default="blade_up")
  ap.add_argument("--stride", type=int, default=2)
  ap.add_argument("--max-solves", type=int, default=6)
  args = ap.parse_args()

  model = IiwaModel.from_mjcf()
  act = ActuatorModel.from_robot(model)
  fns = model.casadi_functions()
  q0 = np.array(BLADE_UP_READY_POS) if args.ready_pose == "blade_up" else default_joint_pos()
  cfg = PlannerConfig(stride=args.stride, max_solves=args.max_solves)
  drop, rng = FruitDropRange(), np.random.default_rng(args.seed)

  n_ok, times, statuses = 0, [], collections.Counter()
  for i in range(args.n):
    p0 = np.array([rng.uniform(*drop.x), rng.uniform(*drop.y), rng.uniform(*drop.z)])
    res = plan_intercept(model, act, q0, np.zeros(7), p0, np.zeros(3), cfg, fns)
    n_ok += res.success
    times.append(res.plan_time)
    for c in res.candidates:
      statuses[c.status] += 1
    best = res.best
    print(
      f"[{i:3d}] fruit {np.round(p0, 2)}  window {res.n_considered:2d}  IK {res.n_ik_ok:2d}  "
      f"vel {res.n_velocity_ok:2d}  solved {len(res.candidates):2d}  "
      + (f"OK T={best.T:.2f}s cost={best.cost:.3g}" if best else "NO PLAN")
      + f"  ({res.plan_time:.1f}s)"
    )
  print(f"\nverified plan for {n_ok}/{args.n} fruits ({n_ok / args.n:.0%})")
  print(f"plan time: mean {np.mean(times):.2f}s  max {np.max(times):.2f}s")
  print("NLP statuses:", dict(statuses))


if __name__ == "__main__":
  main()
