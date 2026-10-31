"""KUKA LBR iiwa 14 as an mjlab entity.

Shared by RL and trajectory optimization. The MJCF in ``assets/iiwa14`` is the
unmodified MuJoCo Menagerie model; everything task-specific (actuators, the
end-effector site, initial pose) is added here in code so the XML stays easy
to diff against upstream.
"""

import math
from pathlib import Path

import mujoco

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
# Link-side inertia each joint sees at READY_JOINT_POS [kg m^2]: the diagonal
# of the joint-space mass matrix without armature.
LINK_INERTIA: dict[str, float] = {
  "joint1": 1.375,
  "joint2": 3.083,
  "joint3": 0.878,
  "joint4": 0.862,
  "joint5": 0.014,
  "joint6": 0.018,
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

# The point on the robot that does the "slicing". For now this is the center of
# the flange (Menagerie's ``attachment_site``). When the blade is added, move
# this to the blade (e.g. its midpoint) -- every consumer reads this name.
EE_SITE_NAME = "ee_site"
EE_SITE_BODY = "link7"
EE_SITE_POS = (0.0, 0.0, 0.045)

# "Ready" pose: flange at ~(0.59, 0, 0.71) m pointing forward, i.e. in the middle
# of the fruit drop zone. (Menagerie's own "home" keyframe puts the flange at
# ~0.29 m height, which is too low to intercept falling fruit early.)
READY_JOINT_POS: dict[str, float] = {
  "joint1": 0.0,
  "joint2": 0.2,
  "joint3": 0.0,
  "joint4": -1.6,
  "joint5": 0.0,
  "joint6": -0.5,
  "joint7": 0.0,
}


def get_iiwa_spec() -> mujoco.MjSpec:
  """Load the Menagerie iiwa and adapt it for mjlab."""
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

  spec.body(EE_SITE_BODY).add_site(
    name=EE_SITE_NAME, pos=EE_SITE_POS, size=(0.01, 0, 0), rgba=(0, 1, 0, 1)
  )
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
