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
    robot.py                    # iiwa EntityCfg: actuators, limits, EE site, ready pose
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
- **Action (6D):** end-effector pose delta → damped least-squares (DLS) IK →
  joint position targets (mjlab `DifferentialIKAction`), at 50 Hz.
- **Observation (privileged):** joint pos/vel, EE pos/quat/vel, fruit position
  relative to the EE, fruit velocity, last action.
- **Cut condition** (`fruitninja_common/slicing.py`): within one control step,
  the blade point passes within `HIT_RADIUS` (= fruit radius) of the fruit
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

## Adding the blade

1. Attach the blade geometry to `link7` (edit `assets/iiwa14/iiwa14.xml`, or
   attach a separate MJCF in `robot.get_iiwa_spec`).
2. Move `EE_SITE_POS` in `robot.py` to the blade's cutting point. Everything
   (IK target, observations, cut condition) reads that one site.
3. If cut direction matters, extend `slicing.check_slice` (e.g. require the
   blade's velocity to be roughly along the blade plane).
