"""Fruit slicing with the KUKA iiwa: the mjlab environment config.

One episode = one fruit. The fruit is dropped from a random point above the
robot and falls straight down; the episode ends when it is sliced, when it
falls below ``MISS_HEIGHT``, or on timeout.

Every field here can be overridden from the CLI, e.g.
  uv run train Fruitninja-Slice-Iiwa --env.scene.num-envs 2048 \
      --env.rewards.ee-fruit-proximity.weight 0.0
"""

from fruitninja_common import slicing
from fruitninja_common.fruit import FruitDropRange, get_fruit_cfg
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

FRUIT = "fruit"
MISS_HEIGHT = 0.15  # [m] fruit center below this height = missed


def _ee() -> SceneEntityCfg:
  """The robot's blade point (a fresh instance per term; mjlab resolves them)."""
  return SceneEntityCfg("robot", site_names=(EE_SITE_NAME,))


def make_slicing_env_cfg(drop: FruitDropRange = FruitDropRange()) -> ManagerBasedRlEnvCfg:
  scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_iiwa_robot_cfg(), FRUIT: get_fruit_cfg()},
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

  # Privileged state: the policy sees the exact fruit position and velocity.
  policy_terms = {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "ee_pos": ObservationTermCfg(func=mdp.ee_pos, params={"asset_cfg": _ee()}),
    "ee_quat": ObservationTermCfg(func=mdp.ee_quat, params={"asset_cfg": _ee()}),
    "ee_lin_vel": ObservationTermCfg(func=mdp.ee_lin_vel, params={"asset_cfg": _ee()}),
    "fruit_pos_rel_ee": ObservationTermCfg(
      func=mdp.fruit_pos_rel_ee, params={"fruit_name": FRUIT, "asset_cfg": _ee()}
    ),
    "fruit_lin_vel": ObservationTermCfg(
      func=mdp.fruit_lin_vel, params={"fruit_name": FRUIT}
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
    "drop_fruit": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": drop.x, "y": drop.y, "z": drop.z},
        "velocity_range": {"x": drop.vx, "y": drop.vy, "z": drop.vz},
        "asset_cfg": SceneEntityCfg(FRUIT),
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fruit_sliced": TerminationTermCfg(
      func=mdp.fruit_sliced,
      params={
        "fruit_name": FRUIT,
        "asset_cfg": _ee(),
        "hit_radius": slicing.HIT_RADIUS,
        "min_blade_speed": slicing.MIN_BLADE_SPEED,
      },
    ),
    "fruit_missed": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": MISS_HEIGHT, "asset_cfg": SceneEntityCfg(FRUIT)},
    ),
  }

  # Reward weights are multiplied by step_dt (0.02 s) by mjlab, so the slice
  # bonus of 500 is worth +10 once, while per-step terms accumulate over the
  # ~25 steps of an episode.
  rewards = {
    # Sparse task reward (proposal Sec. 3.2.2).
    "slice": RewardTermCfg(
      func=mdp.termination_term, weight=500.0, params={"term_name": "fruit_sliced"}
    ),
    # Dense shaping to get learning started. Set weight to 0 for sparse-only.
    "ee_fruit_proximity": RewardTermCfg(
      func=mdp.ee_fruit_proximity,
      weight=1.0,
      params={"std": 0.3, "fruit_name": FRUIT, "asset_cfg": _ee()},
    ),
    "slice_object_reward": RewardTermCfg(
      func=mdp.slice_object_reward,
      weight=1.0,
      params={"miss_std": 0.3, "fruit_name": FRUIT, "asset_cfg": _ee()}
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

  # Logged once per episode under Metrics/ (Metrics/slice_success = success rate).
  metrics = {
    "slice_success": MetricsTermCfg(func=mdp.slice_success, reduce="max"),
    "neg_closest_approach": MetricsTermCfg(func=mdp.closest_approach, reduce="max"),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.5, 0.0, 0.7),
      distance=3.0,
      elevation=-15.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=50,
      njmax=300,
      mujoco=MujocoCfg(timestep=0.005),  # 200 Hz physics
    ),
    decimation=4,  # 50 Hz policy
    # Safety net only: a fruit dropped from 1.7 m reaches MISS_HEIGHT in ~0.56 s.
    episode_length_s=2.0,
  )
