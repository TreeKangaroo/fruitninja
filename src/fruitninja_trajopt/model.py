"""Rigid-body model of the iiwa for trajectory optimization.

Everything is derived from the same sources as the simulator, so the planner
and the mjlab env agree:

* kinematic tree + inertias: parsed from the Menagerie MJCF (``robot.IIWA_XML``),
* PD gains, armature, effort/velocity limits, EE site: ``fruitninja_common.robot``,
* joint limits: MJCF ranges times mjlab's ``soft_joint_pos_limit_factor``.

The recursive Newton-Euler algorithm (``IiwaModel.rnea``) is written with plain
Python scalars and lists, so the very same code runs on floats (tests, IK,
warm starts) and on CasADi ``SX`` symbols (the NLP).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fruitninja_common import robot

GRAVITY = 9.81  # [m/s^2]
NJ = 7
# Blade axis in the EE-site frame. Keep in sync with ``slice_object_reward``'s
# ``blade_axis`` default in fruitninja_rl/tasks/mdp/rewards.py.
BLADE_AXIS = (0.0, 0.0, 1.0)


# --- tiny 3-vector / 3x3 helpers (work for floats and CasADi SX) -------------
def _mat_vec(R, v):
  return [R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3)]


def _matT_vec(R, v):
  return [R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2] for i in range(3)]


def _mat_mat(A, B):
  return [
    [A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3)]
    for i in range(3)
  ]


def _cross(a, b):
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]


def _add(a, b):
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _scale(s, a):
  return [s * a[0], s * a[1], s * a[2]]


def _rot_axis(axis, q, cos, sin):
  """Rodrigues rotation about the constant unit ``axis`` by angle ``q``."""
  c, s = cos(q), sin(q)
  C = 1.0 - c
  x, y, z = axis
  return [
    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ]


def _quat_to_rot(quat):
  w, x, y, z = quat
  n = math.sqrt(w * w + x * x + y * y + z * z)  # MuJoCo normalizes quats
  w, x, y, z = w / n, x / n, y / n, z / n
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ]


def _floats(s: str) -> list[float]:
  return [float(t) for t in s.split()]


# --- MJCF parsing --------------------------------------------------------------
@dataclass(frozen=True)
class Link:
  name: str
  pos: list  # body origin in the parent frame
  rot: list  # body orientation in the parent frame (at q = 0)
  mass: float
  com: list  # center of mass in the body frame
  inertia: list  # 3x3 about the COM, in the body frame
  axis: list  # joint axis in the body frame (unit)
  q_range: tuple[float, float]


def _joint_class_attribs(root: ET.Element) -> dict[str, dict[str, str]]:
  """Resolve <default class=...><joint .../> attributes, with nesting/inheritance."""
  out: dict[str, dict[str, str]] = {}

  def visit(el: ET.Element, inherited: dict[str, str]) -> None:
    cur = dict(inherited)
    joint = el.find("joint")
    if joint is not None:
      cur.update(joint.attrib)
    if el.get("class"):
      out[el.get("class")] = cur
    for child in el.findall("default"):
      visit(child, cur)

  top = root.find("default")
  if top is not None:
    visit(top, {})
  return out


def parse_mjcf_chain(path: Path = robot.IIWA_XML) -> list[Link]:
  root = ET.parse(path).getroot()
  classes = _joint_class_attribs(root)
  body = root.find("worldbody").find("body")  # the fixed base
  childclass = body.get("childclass")
  links: list[Link] = []
  while (child := body.find("body")) is not None:
    childclass = child.get("childclass", childclass)
    joint = child.find("joint")
    attrs = {**classes.get(joint.get("class") or childclass, {}), **joint.attrib}
    axis = _floats(attrs.get("axis", "0 0 1"))
    norm = math.sqrt(sum(a * a for a in axis))
    axis = [a / norm for a in axis]
    lo, hi = _floats(attrs["range"])

    inertial = child.find("inertial")
    assert inertial is not None and inertial.get("diaginertia"), (
      f"{child.get('name')}: only <inertial diaginertia=...> is supported"
    )
    R_in = _quat_to_rot(_floats(inertial.get("quat", "1 0 0 0")))
    d = _floats(inertial.get("diaginertia"))
    # I = R diag(d) R^T
    inertia = [
      [sum(R_in[i][k] * d[k] * R_in[j][k] for k in range(3)) for j in range(3)]
      for i in range(3)
    ]
    links.append(
      Link(
        name=child.get("name"),
        pos=_floats(child.get("pos", "0 0 0")),
        rot=_quat_to_rot(_floats(child.get("quat", "1 0 0 0"))),
        mass=float(inertial.get("mass")),
        com=_floats(inertial.get("pos", "0 0 0")),
        inertia=inertia,
        axis=axis,
        q_range=(lo, hi),
      )
    )
    body = child
  assert len(links) == NJ, f"expected {NJ} links, found {len(links)}"
  return links


# --- the model -------------------------------------------------------------------
class IiwaModel:
  """Serial 7-DoF chain: inverse dynamics, forward kinematics, CasADi functions."""

  def __init__(
    self,
    links: list[Link],
    ee_pos=robot.EE_SITE_POS,
    blade_axis=BLADE_AXIS,
  ):
    assert links[-1].name == robot.EE_SITE_BODY, "EE site must be on the last link"
    self.links = links
    self.ee_pos = [float(x) for x in ee_pos]
    axis = np.asarray(blade_axis, dtype=float)
    self.blade_axis = [float(x) for x in axis / np.linalg.norm(axis)]

  @classmethod
  def from_mjcf(cls, path: Path = robot.IIWA_XML, **kwargs) -> IiwaModel:
    return cls(parse_mjcf_chain(path), **kwargs)

  # -- generic (float or SX) algorithms --------------------------------------
  def rnea(self, q, qd, qdd, trig=(math.cos, math.sin), gravity=GRAVITY):
    """Joint torques tau = M(q) qdd + C(q, qd) qd + g(q) (rigid body, no armature)."""
    cos, sin = trig
    w, al, acc = [0.0] * 3, [0.0] * 3, [0.0, 0.0, gravity]  # base; gravity = -accel
    Rs, Fs, Ns, Cs = [], [], [], []
    for i, L in enumerate(self.links):
      R = _mat_mat(L.rot, _rot_axis(L.axis, q[i], cos, sin))  # child in parent
      w_p = _matT_vec(R, w)  # parent angular velocity, in the child frame
      qd_axis = _scale(qd[i], L.axis)
      w_n = _add(w_p, qd_axis)
      al_n = _add(
        _add(_matT_vec(R, al), _scale(qdd[i], L.axis)), _cross(w_p, qd_axis)
      )
      acc_n = _matT_vec(
        R, _add(_add(acc, _cross(al, L.pos)), _cross(w, _cross(w, L.pos)))
      )
      a_c = _add(_add(acc_n, _cross(al_n, L.com)), _cross(w_n, _cross(w_n, L.com)))
      F = _scale(L.mass, a_c)
      Iw = _mat_vec(L.inertia, w_n)
      N = _add(_mat_vec(L.inertia, al_n), _cross(w_n, Iw))
      Rs.append(R)
      Fs.append(F)
      Ns.append(N)
      Cs.append(L.com)
      w, al, acc = w_n, al_n, acc_n

    tau = [None] * NJ
    f, n = [0.0] * 3, [0.0] * 3  # force/moment exerted by the next link, in its frame
    for i in reversed(range(NJ)):
      if i == NJ - 1:
        f_c, n_c, extra = [0.0] * 3, [0.0] * 3, [0.0] * 3
      else:
        f_c = _mat_vec(Rs[i + 1], f)
        n_c = _mat_vec(Rs[i + 1], n)
        extra = _cross(self.links[i + 1].pos, f_c)
      f = _add(Fs[i], f_c)
      n = _add(_add(_add(Ns[i], _cross(Cs[i], Fs[i])), n_c), extra)
      tau[i] = sum(self.links[i].axis[k] * n[k] for k in range(3))
    return tau

  def fk(self, q, trig=(math.cos, math.sin)):
    """EE-site position and orientation (world frame; robot base at the origin)."""
    cos, sin = trig
    R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    p = [0.0] * 3
    for i, L in enumerate(self.links):
      p = _add(p, _mat_vec(R, L.pos))
      R = _mat_mat(R, _mat_mat(L.rot, _rot_axis(L.axis, q[i], cos, sin)))
    return _add(p, _mat_vec(R, self.ee_pos)), R

  @property
  def shoulder(self) -> np.ndarray:
    """Position of joint 2 (the shoulder) in the base frame."""
    return np.array([0.0, 0.0, self.links[0].pos[2] + self.links[1].pos[2]])

  # -- numeric conveniences --------------------------------------------------
  @staticmethod
  def _f(x):
    return [float(t) for t in x]

  def inverse_dynamics(self, q, qd, qdd, gravity=GRAVITY) -> np.ndarray:
    return np.array(self.rnea(self._f(q), self._f(qd), self._f(qdd), gravity=gravity))

  def mass_matrix(self, q) -> np.ndarray:
    """Joint-space mass matrix without armature."""
    zero = [0.0] * NJ
    M = np.zeros((NJ, NJ))
    for i in range(NJ):
      e = [0.0] * NJ
      e[i] = 1.0
      M[:, i] = self.rnea(self._f(q), zero, e, gravity=0.0)
    return M

  def ee_pos_num(self, q) -> np.ndarray:
    return np.array(self.fk(self._f(q))[0])

  def blade_axis_num(self, q) -> np.ndarray:
    return np.array(_mat_vec(self.fk(self._f(q))[1], self.blade_axis))

  def ee_jacobian_num(self, q, eps: float = 1e-6) -> np.ndarray:
    """Linear-velocity Jacobian of the blade point (3, NJ), central differences."""
    q = np.asarray(q, float)
    cols = [
      (self.ee_pos_num(q + eps * e) - self.ee_pos_num(q - eps * e)) / (2 * eps)
      for e in np.eye(NJ)
    ]
    return np.stack(cols, axis=1)

  def ee_vel_num(self, q, qd, eps: float = 1e-6) -> np.ndarray:
    q, qd = np.asarray(q, float), np.asarray(qd, float)
    return (self.ee_pos_num(q + eps * qd) - self.ee_pos_num(q - eps * qd)) / (2 * eps)

  # -- CasADi ------------------------------------------------------------------
  def casadi_functions(self) -> dict:
    """Symbolic SX functions used by the NLP (built once, called with MX)."""
    import casadi as ca

    q, qd, qdd = (ca.SX.sym(n, NJ) for n in ("q", "qd", "qdd"))
    trig = (ca.cos, ca.sin)
    tau = ca.vertcat(*self.rnea(q, qd, qdd, trig))
    p, R = self.fk(q, trig)
    p = ca.vertcat(*p)
    axis = ca.vertcat(*_mat_vec(R, self.blade_axis))
    vel = ca.mtimes(ca.jacobian(p, q), qd)
    return {
      "inverse_dynamics": ca.Function("inverse_dynamics", [q, qd, qdd], [tau]),
      "ee_pos": ca.Function("ee_pos", [q], [p]),
      "blade_axis": ca.Function("blade_axis", [q], [axis]),
      "ee_vel": ca.Function("ee_vel", [q, qd], [vel]),
    }


@dataclass(frozen=True)
class ActuatorModel:
  """PD actuators + limits, exactly as configured for the simulator."""

  kp: np.ndarray
  kd: np.ndarray
  armature: np.ndarray
  effort: np.ndarray
  vel_limit: np.ndarray
  q_lo: np.ndarray
  q_hi: np.ndarray

  @classmethod
  def from_robot(cls, model: IiwaModel, soft_factor: float | None = None):
    names = robot.JOINT_NAMES
    if soft_factor is None:
      soft_factor = robot.ARTICULATION.soft_joint_pos_limit_factor
    lo = np.array([L.q_range[0] for L in model.links])
    hi = np.array([L.q_range[1] for L in model.links])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * soft_factor
    arr = lambda d: np.array([d[n] for n in names], dtype=float)  # noqa: E731
    return cls(
      kp=arr(robot.STIFFNESS),
      kd=arr(robot.DAMPING),
      armature=arr(robot.ARMATURE),
      effort=arr(robot.EFFORT_LIMITS),
      vel_limit=arr(robot.VELOCITY_LIMITS),
      q_lo=mid - half,
      q_hi=mid + half,
    )

  def pd_torque(self, q, qd, u):
    return self.kp * (u - q) - self.kd * qd


def default_joint_pos() -> np.ndarray:
  """The shared READY pose (flange pointing forward)."""
  return np.array([robot.READY_JOINT_POS[n] for n in robot.JOINT_NAMES])


# A ready pose with the blade already pointing up (axis . z ~ 0.99), blade point
# at ~(0.59, 0, 0.85). From ``READY_JOINT_POS`` the wrist must swing ~90 deg to
# get the blade vertical, which the joint velocity limits do not allow in the
# ~0.4 s window for most of the drop range (see README). Start here instead.
BLADE_UP_READY_POS = (0.0, 0.43, 0.0, -1.17, 0.0, -1.52, 0.0)
