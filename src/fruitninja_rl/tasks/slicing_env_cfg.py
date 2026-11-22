"""Fruit slicing with the KUKA iiwa: the mjlab environment config.

One episode = ``EPISODE_LENGTH_S`` seconds of a fruit stream. Fruit are dropped
one after another (random gap, ``SPAWN_INTERVAL_S``) from random points above
the robot, each on a randomly chosen trajectory (``TRAJECTORIES`` in fruit.py:
straight down, or drifting in x). The episode only ends on timeout; the robot
should slice as many fruit as it can. Sliced fruit turn green and keep falling;
every fruit disappears when it reaches the floor (see ``mdp/fruit_pool.py``).

``make_slicing_env_cfg(multi_fruit=False)`` is the single-fruit variant: one
fruit per episode, which ends as soon as the fruit is sliced or reaches the floor.

Every field here can be overridden from the CLI, e.g.
  uv run train Fruitninja-Slice-Iiwa --env.scene.num-envs 2048 \
      --env.rewards.ee-fruit-proximity.weight 0.0
"""

from fruitninja_common.fruit import (
  NUM_FRUIT_SLOTS,
  SLICED_COLOR,
  SPAWN_INTERVAL_S,
  TRAJECTORIES,
  fruit_name,
  get_fruit_cfg,
  sliced_fruit_name,
)
from fruitninja_common.robot import EE_SITE_NAME, VELOCITY_LIMITS, get_iiwa_robot_cfg
from fruitninja_rl.tasks import mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

EPISODE_LENGTH_S = 10.0  # multi-fruit
SINGLE_FRUIT_EPISODE_LENGTH_S = 2.0  # safety net: a fruit lands in ~0.6 s
FRUIT_NAMES = tuple(fruit_name(i) for i in range(NUM_FRUIT_SLOTS))
SLICED_FRUIT_NAMES = tuple(sliced_fruit_name(i) for i in range(NUM_FRUIT_SLOTS))


def _ee() -> SceneEntityCfg:
  """The robot's blade point (a fresh instance per term; mjlab resolves them)."""
  return SceneEntityCfg("robot", site_names=(EE_SITE_NAME,))


def make_slicing_env_cfg(
  spawn_interval_s: tuple[float, float] = SPAWN_INTERVAL_S,
  multi_fruit: bool = True,
) -> ManagerBasedRlEnvCfg:
  scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={
      "robot": get_iiwa_robot_cfg(),
      **{name: get_fruit_cfg() for name in FRUIT_NAMES},
      **{name: get_fruit_cfg(SLICED_COLOR) for name in SLICED_FRUIT_NAMES},
    },
    num_envs=1,
    env_spacing=2.0,
  )

  # The policy outputs a 6D end-effector pose delta (dx, dy, dz, rotation
  # vector); mjlab's damped-least-squares IK turns it into joint targets every
  # physics substep. Actions are clipped to [-1, 1] by the runner, so the
  # largest commanded step is 5 cm / 0.1 rad per 20 ms policy step.
  actions = {
    "ee_pose": DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      frame_type="site",
      frame_name=EE_SITE_NAME,
      use_relative_mode=True,
      delta_pos_scale=0.05,
      delta_ori_scale=0.1,
      max_dq=0.1,
      joint_limit_weight=1.0,
      # The iiwa is redundant (7 DoF for a 6D target). A weak pull towards the
      # ready pose keeps the elbow from drifting in the null space.
      posture_weight=0.05,
    )
  }

  # Privileged state: the policy sees the exact position and velocity of every
  # unsliced fruit in the air, lowest first (see mdp.upcoming_fruit).
  policy_terms = {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "ee_pos": ObservationTermCfg(func=mdp.ee_pos, params={"asset_cfg": _ee()}),
    "ee_quat": ObservationTermCfg(func=mdp.ee_quat, params={"asset_cfg": _ee()}),
    "ee_lin_vel": ObservationTermCfg(func=mdp.ee_lin_vel, params={"asset_cfg": _ee()}),
    "upcoming_fruit": ObservationTermCfg(
      func=mdp.upcoming_fruit,
      params={"asset_cfg": _ee(), "num_fruits": NUM_FRUIT_SLOTS},
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  observations = {
    # No observation noise yet; add `noise=` to terms to study robustness.
    "actor": ObservationGroupCfg(policy_terms, enable_corruption=False),
    "critic": ObservationGroupCfg({**policy_terms}, enable_corruption=False),
  }

  events = {
    # Places each env's (fixed-base) robot at its env origin.
    "reset_robot_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Clears the env's fruit and drops the first one at reset (the only one in
    # single-fruit mode)...
    "fruit_pool": EventTermCfg(
      func=mdp.FruitPool,
      mode="reset",
      params={
        "fruit_names": FRUIT_NAMES,
        "sliced_fruit_names": SLICED_FRUIT_NAMES,
        "trajectories": dict(TRAJECTORIES),
        "asset_cfg": _ee(),
      },
    ),
  }
  if multi_fruit:
    # ...then the next one after a random gap from the previous drop.
    events["spawn_fruit"] = EventTermCfg(
      func=mdp.spawn_fruit, mode="interval", interval_range_s=spawn_interval_s
    )

  # Multi-fruit: slicing does not end the episode, only the clock does.
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  if not multi_fruit:
    terminations["fruit_resolved"] = TerminationTermCfg(
      func=mdp.fruits_resolved, params={"num_fruits": 1}
    )

  # Reward weights are multiplied by step_dt (0.02 s) by mjlab, so the slice
  # bonus of 500 is worth +10 per fruit, while per-step terms accumulate over
  # the steps of an episode (500 in multi-fruit mode).
  rewards = {
    # Sparse task reward (proposal Sec. 3.2.2): once per fruit sliced.
    "slice": RewardTermCfg(func=mdp.slices_this_step, weight=500.0),
    # Penalty for every fruit that reaches the floor unsliced (-5 each).
    "fruit_missed": RewardTermCfg(func=mdp.fruits_missed_this_step, weight=-250.0),
    # Dense shaping: reach a fruit, then slice it (reaching * (1 + slicing)),
    # scored for every fruit in the air; the best one counts.
    # Set weight to 0 for sparse-only.
    "staged_slice_reward": RewardTermCfg(
      func=mdp.staged_slice_reward,
      weight=1.0,
      params={
        "reaching_std": 0.3,
        "miss_std": 0.3,
        "asset_cfg": _ee(),
      },
    ),
    # The two parts of staged_slice_reward on their own. Off by default (they
    # are already inside it); give one a weight to try it separately.
    "ee_fruit_proximity": RewardTermCfg(
      func=mdp.ee_fruit_proximity,
      weight=0.0,
      params={"std": 0.3, "asset_cfg": _ee()},
    ),
    "slice_object_reward": RewardTermCfg(
      func=mdp.slice_object_reward,
      weight=0.0,
      params={"miss_std": 0.3, "asset_cfg": _ee()},
    ),
    # Motion-quality regularizers.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_limits": RewardTermCfg(
      func=mdp.joint_vel_limits,
      weight=-1.0,
      params={"limits": VELOCITY_LIMITS, "asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  # Logged once per episode under Episode_Metrics/. Counts are per episode;
  # slice_rate = sliced / (sliced + missed) is the headline success number.
  metrics = {
    "fruits_spawned": MetricsTermCfg(func=mdp.fruits_spawned, reduce="last"),
    "fruits_sliced": MetricsTermCfg(func=mdp.fruits_sliced, reduce="last"),
    "fruits_missed": MetricsTermCfg(func=mdp.fruits_missed, reduce="last"),
    "slice_rate": MetricsTermCfg(func=mdp.slice_rate, reduce="last"),
    "peak_blade_speed": MetricsTermCfg(func=mdp.blade_speed, reduce="max"),
    "peak_fruits_in_air": MetricsTermCfg(func=mdp.fruits_in_air, reduce="max"),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    # Track the robot, not a fixed world point: env 0 sits at a grid offset that
    # is far from the world origin when num_envs is large, so a WORLD camera
    # films an empty floor in training videos.
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=4.0,
      elevation=-15.0,
      azimuth=135.0,
      width=640,
      height=480,
    ),
    sim=SimulationCfg(
      nconmax=50,
      njmax=300,
      mujoco=MujocoCfg(timestep=0.005),  # 200 Hz physics
    ),
    decimation=4,  # 50 Hz policy
    episode_length_s=EPISODE_LENGTH_S if multi_fruit else SINGLE_FRUIT_EPISODE_LENGTH_S,
  )
