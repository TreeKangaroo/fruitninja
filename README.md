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
    fruit.py                    # fruit sphere + TRAJECTORIES (where/how fruit spawn)
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

Two variants share observation/action spaces (checkpoints transfer between
them). Pick one with `--fruit-mode multi|single` on `train`/`play` (default
`multi`), or by task id: `Fruitninja-Slice-Iiwa` (multi) and
`Fruitninja-Slice-Iiwa-Single`. **Single:** one fruit per episode, which ends as
soon as the fruit is sliced or lands (2 s cap). The rest of this section
describes multi.

- **Episode:** a 10 s stream of fruit (500 steps at 50 Hz). The first fruit
  drops at reset and each next one follows after a random 0.15–0.6 s gap
  (`SPAWN_INTERVAL_S` in `fruit.py`), about 27 per episode, with up to
  `NUM_FRUIT_SLOTS` = 4 in the air at once (a spawn is skipped while all 4
  slots are busy). Each takes ~0.55 s to fall. Slicing does **not** end the
  episode; only the clock does.
- **Trajectories** (both variants): each fruit independently picks an entry of
  `TRAJECTORIES` in `fruit.py`, in proportion to its `weight` (equal by default):

  | name | release x [m] | vx [m/s] |
  |---|---|---|
  | `straight` | 0.45–0.70 | 0 |
  | `xvel` | 0.79–0.885 | -0.75 to -0.50 |

  Both release at y ±0.3 m, z 1.4–1.7 m and cross blade height (z = 0.70 m)
  within x = 0.45–0.70 m. To add a trajectory, add an entry to the dict. To
  change the mix for a training run:
  `--env.events.fruit-pool.params.trajectories.xvel.weight 2`.
- **Fruit lifecycle** (`tasks/mdp/fruit_pool.py`): red while whole, green once
  sliced (it keeps falling), and it disappears when it reaches the floor. Each
  env has `NUM_FRUIT_SLOTS` reusable slots, and each slot has a red and a green
  body; cutting a fruit hands its state to the green one. The viewers need no
  special support.
- **Robot:** iiwa 14 with a rigid toy blade on the flange (see below). It starts
  in a "ready" pose with the blade level across the drop zone, edge up.
- **Action (6D):** blade-center pose delta → damped least-squares (DLS) IK →
  joint position targets (mjlab `DifferentialIKAction`), at 50 Hz.
- **Observation (privileged):** joint pos/vel, EE pos/quat/vel, every
  unsliced fruit in the air, lowest first (position relative to the EE,
  velocity, valid flag; zeros for empty slots), last action.
- **Cut condition** (`fruitninja_common/slicing.py`): within one control step,
  the blade *center* passes within `HIT_RADIUS` (= fruit radius) of the fruit
  center while moving at ≥ `MIN_BLADE_SPEED` (0.5 m/s). The check uses closest
  approach over the step, so fast pass-throughs between steps are not missed. The fruit
  does not collide with the robot.
- **Rewards:** +10 per fruit sliced and -5 per fruit that hits the floor
  unsliced (both sparse), `staged_slice_reward` shaping (scored for every
  fruit in the air, the best one counts), and penalties on action rate, joint
  position limits and joint velocity limits.
- **Logged metrics** (`Episode_Metrics/*`, per episode): `slice_rate`
  (sliced / (sliced + missed)), `fruits_sliced`, `fruits_missed`,
  `fruits_spawned`, `peak_fruits_in_air`, `peak_blade_speed`.

## Usage

```bash
# Sanity check (CPU is fine): idle policy never slices, scripted chaser does (one full episode each).
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
