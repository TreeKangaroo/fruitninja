"""KUKA LBR iiwa 14 as an mjlab entity.

Shared by RL and trajectory optimization. The MJCF in ``assets/iiwa14`` is the
unmodified MuJoCo Menagerie model; everything task-specific (actuators, the
blade, initial pose) is added here in code so the XML stays easy
to diff against upstream.
"""

import math
from pathlib import Path

import mujoco

from fruitninja_common import blade
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

IIWA_XML: Path = Path(__file__).parent / "assets" / "iiwa14" / "iiwa14.xml"

JOINT_NAMES: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))

# Hardware limits from the LBR iiwa 14 R820 datasheet.
EFFORT_LIMITS: dict[str, float] = {  # [N*m]
  "joint1": 320.0,
  "joint2": 320.0,
  "joint3": 176.0,
  "joint4": 176.0,
  "joint5": 110.0,
  "joint6": 40.0,
  "joint7": 40.0,
}
VELOCITY_LIMITS: dict[str, float] = {  # [rad/s]
  name: math.radians(deg_s)
  for name, deg_s in zip(JOINT_NAMES, (85, 85, 100, 75, 130, 135, 135), strict=True)
}

# --- Actuator model -----------------------------------------------------------
# Menagerie uses kp=2000, kd=200 on every joint and no rotor inertia. That is
# unstable in practice: the wrist links are so light (joint7: 0.001 kg m^2)
# that when the torque limit saturates the joint oscillates at >100 rad/s.
# Instead we follow mjlab's convention (see its YAM/G1 robots):
#   1. add reflected rotor inertia (armature), which dominates on geared arms;
#   2. derive each joint's PD gains from its effective inertia so every joint
#      behaves like a critically damped 2nd-order system at NATURAL_FREQ.

# Reflected rotor inertia [kg m^2]. ESTIMATES (KUKA does not publish these);
# replace if better numbers turn up.
ARMATURE: dict[str, float] = {
  "joint1": 0.4,
  "joint2": 0.4,
  "joint3": 0.2,
  "joint4": 0.2,
  "joint5": 0.1,
  "joint6": 0.05,
  "joint7": 0.05,
}
# Link-side inertia each joint sees at READY_JOINT_POS, blade included
# [kg m^2]: the diagonal of the joint-space mass matrix without armature.
# Recompute if the blade or the ready pose changes.
LINK_INERTIA: dict[str, float] = {
  "joint1": 3.715,
  "joint2": 4.124,
  "joint3": 0.573,
  "joint4": 0.860,
  "joint5": 0.074,
  "joint6": 0.069,
  "joint7": 0.001,
}
NATURAL_FREQ = 2.0 * math.pi * 10.0  # [rad/s] 10 Hz position bandwidth
DAMPING_RATIO = 1.0  # critically damped


def _effective_inertia(name: str) -> float:
  return LINK_INERTIA[name] + ARMATURE[name]


STIFFNESS: dict[str, float] = {
  n: _effective_inertia(n) * NATURAL_FREQ**2 for n in JOINT_NAMES
}
DAMPING: dict[str, float] = {
  n: 2.0 * DAMPING_RATIO * _effective_inertia(n) * NATURAL_FREQ for n in JOINT_NAMES
}

# The blade (see blade.py) is mounted rigidly on the flange: link7's
# ``attachment_site`` position, axes aligned with link7.
FLANGE_BODY = "link7"
FLANGE_POS = (0.0, 0.0, 0.045)

# The "end effector" used for IK control and observations: the blade center.
# Every consumer (action, observations, rewards, cut check) reads this name.
EE_SITE_NAME = blade.BLADE_CENTER_SITE

# "Ready" pose: the blade lies level across the middle of the fruit drop zone,
# edge pointing straight up: blade center at ~(0.55, 0, 0.70) m, cutting edge
# spanning y = -0.2..0.2 m at z ~0.725 m. The arm reaches in from the -y side.
# Found by IK; joint6 is at ~80% of its range, which is unavoidable for a
# level blade with this mount.
READY_JOINT_POS: dict[str, float] = {
  "joint1": -0.944,
  "joint2": 1.348,
  "joint3": -0.853,
  "joint4": 0.814,
  "joint5": -0.257,
  "joint6": -1.707,
  "joint7": -0.897,
}


def get_iiwa_spec() -> mujoco.MjSpec:
  """Load the Menagerie iiwa, adapt it for mjlab, and mount the blade."""
  spec = mujoco.MjSpec.from_file(str(IIWA_XML))

  # mjlab creates actuators from EntityArticulationInfoCfg, and its own
  # initial-state keyframe, so drop the ones that ship with the XML.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for key in list(spec.keys):
    spec.delete(key)
  # The integrator is set globally by mjlab's SimulationCfg (implicitfast);
  # resetting it here avoids an "attach conflict" warning.
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  # One light per robot would be replicated across every parallel env.
  for light in list(spec.lights):
    spec.delete(light)

  blade.attach_blade(spec.body(FLANGE_BODY), pos=FLANGE_POS)
  return spec


ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,),
      stiffness=STIFFNESS[name],
      damping=DAMPING[name],
      effort_limit=EFFORT_LIMITS[name],
      armature=ARMATURE[name],
    )
    for name in JOINT_NAMES
  ),
  soft_joint_pos_limit_factor=0.95,
)


def get_iiwa_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_iiwa_spec,
    articulation=ARTICULATION,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      joint_pos=READY_JOINT_POS,
      joint_vel={".*": 0.0},
    ),
  )
