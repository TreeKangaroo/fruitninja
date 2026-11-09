# Fruit Ninja Robot

KUKA LBR iiwa 14 slicing falling fruit in simulation: comparing trajectory
optimization with reinforcement learning ([proposal](FruitNinjaProposal.pdf)).
Built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + rsl_rl PPO).

## Setup

```bash
uv sync    # creates .venv with mjlab, torch, mujoco-warp, ...
```

## Layout

```
src/
  fruitninja_common/            # shared by RL and trajopt
    assets/iiwa14/              # unmodified MuJoCo Menagerie kuka_iiwa_14
    robot.py                    # iiwa EntityCfg: actuators, limits, ready pose
    blade.py                    # toy blade rigidly mounted on the flange
    fruit.py                    # fruit sphere + FruitDropRange (where fruit spawns)
    slicing.py                  # THE cut condition (numpy or torch)
  fruitninja_rl/
    tasks/
      __init__.py               # registers "Fruitninja-Slice-Iiwa" with mjlab
      slicing_env_cfg.py        # the whole MDP: scene, actions, obs, rewards, ...
      rl_cfg.py                 # PPO hyperparameters
      mdp/                      # task-specific obs/reward/termination/metric terms
    scripts/
      train.py, play.py         # thin wrappers around mjlab's train/play
      smoke_test.py             # scripted-policy sanity check
```

The task is found by mjlab through the `mjlab.tasks` entry point in
`pyproject.toml`, so mjlab's own CLIs work too (`uv run list-envs`,
`uv run train Fruitninja-Slice-Iiwa`, `uv run play Fruitninja-Slice-Iiwa`).

## The RL task

- **Episode:** one fruit. It is dropped from rest at a random point in
  `FruitDropRange` (x 0.45–0.7 m, y ±0.3 m, z 1.4–1.7 m) and falls straight
  down. The episode ends when the fruit is sliced or drops below 0.15 m (about 0.5 s).
- **Robot:** iiwa 14 with a rigid toy blade on the flange (see below). It starts
  in a "ready" pose with the blade level across the drop zone, edge up.
- **Action (6D):** blade-center pose delta → damped least-squares (DLS) IK →
  joint position targets (mjlab `DifferentialIKAction`), at 50 Hz.
- **Observation (privileged):** joint pos/vel, EE pos/quat/vel, fruit position
  relative to the EE, fruit velocity, last action.
- **Cut condition** (`fruitninja_common/slicing.py`): within one control step,
  the blade *center* passes within `HIT_RADIUS` (= fruit radius) of the fruit
  center while moving at ≥ `MIN_BLADE_SPEED` (0.5 m/s). The check uses closest
  approach over the step, so fast pass-throughs between steps are not missed. The fruit
  does not collide with the robot.
- **Rewards:** +10 per slice (sparse), EE–fruit proximity shaping, and penalties
  on action rate, joint position limits and joint velocity limits.
- **Logged success rate:** `Episode_Metrics/slice_success`.

## Usage

```bash
# Sanity check (CPU is fine): idle policy never slices, scripted chaser does.
uv run python -m fruitninja_rl.scripts.smoke_test

# Train (GPU). Any config field can be overridden from the CLI; see --help.
uv run python -m fruitninja_rl.scripts.train --env.scene.num-envs 4096
uv run python -m fruitninja_rl.scripts.train --env.scene.num-envs 64 --gpu-ids None  # CPU debug

# Look at the environment with no policy (zero action = arm holds its pose)
uv run python -m fruitninja_rl.scripts.play --viewer viser

# Watch a trained policy in the browser (http://localhost:8080)
uv run python -m fruitninja_rl.scripts.play --viewer viser \
    --checkpoint-file logs/rsl_rl/iiwa_slice/<run>/model_<N>.pt

tensorboard --logdir logs/rsl_rl   # or set logger="wandb" in rl_cfg.py
```

## The blade

`fruitninja_common/blade.py` adds a rigid (jointless) body `blade` to `link7`:
a 6 cm cylindrical hilt, then a 40 × 5 × 0.5 cm box along the flange axis
(0.4 kg in total). It collides with the floor and the arm but not with the
fruit. A red, visual-only strip marks the cutting edge, which is the long side
at +x in the blade frame. Everything is in `mjData`:

| what | name | read with |
|---|---|---|
| blade frame | body `blade` | `data.xpos`, `data.xmat` / entity `body_*` |
| blade center (= IK/obs end effector) | site `blade_center` | `data.site_xpos` |
| cutting edge endpoints | sites `blade_edge_base`, `blade_edge_tip` | `data.site_xpos` |

Names and dimensions are constants in `blade.py`. In mjlab the names are
prefixed with the entity name (e.g. `robot/blade_edge_tip`). If you change the
blade or the ready pose, recompute `LINK_INERTIA` in `robot.py` (it sets the
PD gains).
