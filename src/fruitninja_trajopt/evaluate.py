"""Plan with trajectory optimization, replay in the mjlab env, judge with ``fruit_sliced``.

  uv run python -m fruitninja_trajopt.evaluate --num-fruits 32 --device cpu

The env is the RL task's (same physics, fruit drops, and ``fruit_sliced``
termination = ``slicing.check_slice``), except that the action is the joint
position target, so a plan's ``u`` is sent as-is. Each parallel env is one
fruit: reset -> read the exact privileged state -> plan -> replay open loop.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from fruitninja_common.robot import JOINT_NAMES
from fruitninja_rl.tasks.slicing_env_cfg import MISS_HEIGHT, make_slicing_env_cfg
from fruitninja_trajopt.model import (
  BLADE_UP_READY_POS,
  ActuatorModel,
  IiwaModel,
  default_joint_pos,
)
from fruitninja_trajopt.planner import PlannerConfig, plan_intercept
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg

NO_PLAN_STEPS = 40  # upper bound on episode length in steps (fruit lands in ~28)


def make_trajopt_env_cfg(num_envs: int, ready_pose, reset_noise: float):
  cfg = make_slicing_env_cfg()
  cfg.scene.num_envs = num_envs
  # Absolute joint-position targets (no default offset), straight to the PD actuators.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,
      use_default_offset=False,
    )
  }
  cfg.scene.entities["robot"].init_state.joint_pos = dict(
    zip(JOINT_NAMES, (float(x) for x in ready_pose), strict=True)
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-reset_noise, reset_noise)
  return cfg


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-fruits", type=int, default=16)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--ready-pose", choices=("blade_up", "current"), default="blade_up")
  ap.add_argument("--reset-noise", type=float, default=0.0, help="joint offset range [rad]")
  ap.add_argument("--stride", type=int, default=2)
  ap.add_argument("--max-solves", type=int, default=6)
  ap.add_argument("--out", default=None, help="write per-fruit results as JSON")
  args = ap.parse_args()
  torch.manual_seed(args.seed)

  ready = BLADE_UP_READY_POS if args.ready_pose == "blade_up" else tuple(default_joint_pos())
  n = args.num_fruits
  env = ManagerBasedRlEnv(make_trajopt_env_cfg(n, ready, args.reset_noise), device=args.device)
  env.reset()

  # --- read the privileged state (robot base frame = env origin) ------------------
  robot, fruit = env.scene["robot"], env.scene["fruit"]
  origins = env.scene.env_origins
  q0 = robot.data.joint_pos.cpu().numpy()
  qd0 = robot.data.joint_vel.cpu().numpy()
  p0 = (fruit.data.root_link_pos_w - origins).cpu().numpy()
  v0 = fruit.data.root_link_lin_vel_w.cpu().numpy()
  print(f"fruit z range at reset: {p0[:, 2].min():.2f}-{p0[:, 2].max():.2f} m (expect 1.4-1.7)")

  # --- plan ------------------------------------------------------------------------------
  model = IiwaModel.from_mjcf()
  act = ActuatorModel.from_robot(model)
  fns = model.casadi_functions()
  cfg = PlannerConfig(z_floor=MISS_HEIGHT + 0.03, stride=args.stride, max_solves=args.max_solves)
  plans, plan_times = [], []
  for i in range(n):
    res = plan_intercept(model, act, q0[i], qd0[i], p0[i], v0[i], cfg, fns)
    plans.append(res)
    plan_times.append(res.plan_time)
    tag = f"T={res.best.T:.2f}s" if res.best else "no plan"
    print(f"[{i:3d}] {tag:12s} ({res.plan_time:.1f}s)")

  # --- replay open loop (no plan -> the arm just holds its pose) -----------------------
  actions = np.tile(q0[:, None, :], (1, NO_PLAN_STEPS, 1))
  for i, res in enumerate(plans):
    if res.best is not None:
      a = res.best.actions_abs()
      actions[i, : len(a)] = a
      actions[i, len(a) :] = a[-1]
  actions_t = torch.as_tensor(actions, dtype=torch.float32, device=env.device)

  sliced = torch.zeros(n, dtype=torch.bool, device=env.device)
  finished = torch.zeros_like(sliced)
  closest = torch.full((n,), float("inf"), device=env.device)
  slice_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
  tm = env.termination_manager
  cut_term = tm.get_term_cfg("fruit_sliced").func
  for k in range(NO_PLAN_STEPS):
    env.step(actions_t[:, k])
    s, m, t = (tm.get_term(x) for x in ("fruit_sliced", "fruit_missed", "time_out"))
    active = ~finished
    if k >= 1:  # the cut term has no previous step to compare to on step 0
      closest = torch.where(active, torch.minimum(closest, cut_term.closest_distance), closest)
    newly = active & s
    slice_step = torch.where(newly, torch.full_like(slice_step, k + 1), slice_step)
    sliced |= newly
    finished |= s | m | t
    if bool(finished.all()):
      break

  # --- report -----------------------------------------------------------------------------
  planned = np.array([r.success for r in plans])
  sim = sliced.cpu().numpy()
  summary = {
    "num_fruits": n,
    "ready_pose": args.ready_pose,
    "plan_found": float(planned.mean()),
    "sim_slice_rate": float(sim.mean()),
    "sim_slice_rate_given_plan": float(sim[planned].mean()) if planned.any() else float("nan"),
    "mean_plan_time_s": float(np.mean(plan_times)),
    "max_plan_time_s": float(np.max(plan_times)),
    "mean_nlp_time_s": float(np.mean([r.nlp_time for r in plans])),
    "mean_ik_time_s": float(np.mean([r.ik_time for r in plans])),
  }
  print(json.dumps(summary, indent=2))
  if args.out:
    per_fruit = [
      {
        "fruit_p0": p0[i].tolist(),
        "plan_found": bool(planned[i]),
        "T": plans[i].best.T if plans[i].best else None,
        "plan_check": plans[i].best.check if plans[i].best else None,
        "sim_sliced": bool(sim[i]),
        "sim_slice_step": int(slice_step[i]),
        "sim_closest_distance": float(closest[i]),
        "plan_time_s": plan_times[i],
      }
      for i in range(n)
    ]
    with open(args.out, "w") as f:
      json.dump({"summary": summary, "fruits": per_fruit}, f, indent=2)
  env.close()


if __name__ == "__main__":
  main()
