"""Visualize trajectory-optimized interception in the shared mjlab scene."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from fruitninja_trajopt.evaluate import make_trajopt_env_cfg
from fruitninja_trajopt.model import (
    BLADE_UP_READY_POS,
    ActuatorModel,
    IiwaModel,
    default_joint_pos,
)
from fruitninja_trajopt.planner import PlannerConfig, plan_intercept
from mjlab.envs import ManagerBasedRlEnv


class TrajoptPolicy:
    """Plan after each reset and replay absolute joint-position targets."""

    def __init__(self, env, cfg: PlannerConfig):
        self.env = env
        self.cfg = cfg
        self.model = IiwaModel.from_mjcf()
        self.act = ActuatorModel.from_robot(self.model)
        self.fns = self.model.casadi_functions()
        self.actions: dict[int, np.ndarray] = {}
        self.last_step = np.full(env.num_envs, -1, dtype=int)

    def __call__(self, obs):
        del obs
        env = self.env
        steps = env.episode_length_buf.detach().cpu().numpy()
        robot, fruit = env.scene["robot"], env.scene["fruit"]
        targets = []

        for i, step in enumerate(steps):
            if i not in self.actions or step == 0 or step < self.last_step[i]:
                q = robot.data.joint_pos[i].detach().cpu().numpy().copy()
                v = robot.data.joint_vel[i].detach().cpu().numpy().copy()
                p = (
                    fruit.data.root_link_pos_w[i] - env.scene.env_origins[i]
                ).detach().cpu().numpy()
                fv = fruit.data.root_link_lin_vel_w[i].detach().cpu().numpy()

                result = plan_intercept(
                    self.model, self.act, q, v, p, fv, self.cfg, self.fns
                )
                self.actions[i] = (
                    result.best.actions_abs()
                    if result.best is not None
                    else q[None, :]
                )
                tag = (
                    f"intercept at {result.best.T:.2f}s"
                    if result.best is not None
                    else "no plan; holding pose"
                )
                print(
                    f"[env {i}] {tag} (planning {result.plan_time:.2f}s)",
                    flush=True,
                )

            actions = self.actions[i]
            targets.append(actions[min(int(step), len(actions) - 1)])
            self.last_step[i] = step

        return torch.as_tensor(
            np.stack(targets), dtype=torch.float32, device=env.device
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--viewer", choices=("auto", "viser", "native"), default="auto")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ready-pose", choices=("blade_up", "current"), default="blade_up"
    )
    ap.add_argument("--reset-noise", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-solves", type=int, default=6)
    args = ap.parse_args()

    if (
        args.num_envs < 1
        or args.stride < 1
        or args.max_solves < 1
        or args.reset_noise < 0
    ):
        ap.error(
            "num-envs, stride and max-solves must be positive; "
            "reset-noise must be nonnegative"
        )

    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    torch.manual_seed(args.seed)
    ready = (
        BLADE_UP_READY_POS
        if args.ready_pose == "blade_up"
        else tuple(default_joint_pos())
    )
    cfg = make_trajopt_env_cfg(args.num_envs, ready, args.reset_noise)
    env = ManagerBasedRlEnv(cfg, device=args.device)

    try:
        policy = TrajoptPolicy(
            env, PlannerConfig(stride=args.stride, max_solves=args.max_solves)
        )
        viewer = args.viewer
        if viewer == "auto":
            has_display = os.environ.get("DISPLAY") or os.environ.get(
                "WAYLAND_DISPLAY"
            )
            viewer = "native" if has_display else "viser"

        print(
            "Planning runs synchronously at each fruit reset; "
            "the viewer may pause while solving."
        )
        viewer_cls = NativeMujocoViewer if viewer == "native" else ViserPlayViewer
        viewer_cls(env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()