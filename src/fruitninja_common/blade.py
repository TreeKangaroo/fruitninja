"""A toy blade rigidly mounted on the iiwa flange.

Geometry, in the blade body frame (origin at the flange, axes = link7's axes):

             +x (edge side)
              ^
              |   cutting edge (red strip), from EDGE_BASE to EDGE_TIP
              |   ===========================================
   flange |hilt|          blade (thin box)                  |  --> +z (along the blade)
              |   -------------------------------------------
              |   spine

  * +z points out of the flange, along the blade.
  * The cutting edge is the long side at x = +WIDTH/2 (y = 0, mid-thickness).
  * The blade's flat faces are normal to y; the blade is THICKNESS thick in y.

Sites (names are module constants, read them from mjData via site_xpos):
  * BLADE_CENTER_SITE  geometric center of the blade box
  * EDGE_BASE_SITE     cutting edge, end nearest the flange
  * EDGE_TIP_SITE      cutting edge, far end
"""

import mujoco

BLADE_BODY_NAME = "blade"
BLADE_GEOM_NAME = "blade_geom"
BLADE_CENTER_SITE = "blade_center"
EDGE_BASE_SITE = "blade_edge_base"
EDGE_TIP_SITE = "blade_edge_tip"

# Dimensions [m]. A light, sword-like prop.
HILT_LENGTH = 0.06  # mount between flange and blade
HILT_RADIUS = 0.02
LENGTH = 0.30  # along z
WIDTH = 0.05  # along x, spine to edge
THICKNESS = 0.005  # along y
# Mass [kg]. Inertia is computed by MuJoCo from the geom shapes.
HILT_MASS = 0.1
BLADE_MASS = 0.3

_STEEL = (0.75, 0.75, 0.8, 1.0)
_DARK = (0.15, 0.15, 0.15, 1.0)
_EDGE = (0.9, 0.1, 0.1, 1.0)


def attach_blade(parent: mujoco.MjsBody, pos: tuple[float, float, float]) -> None:
  """Add the blade as a rigid (jointless) child body of ``parent`` at ``pos``."""
  body = parent.add_body(name=BLADE_BODY_NAME, pos=pos)

  # Hilt: a short cylinder from the flange to the blade.
  body.add_geom(
    name="blade_hilt",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(HILT_RADIUS, HILT_LENGTH / 2, 0),
    pos=(0, 0, HILT_LENGTH / 2),
    mass=HILT_MASS,
    rgba=_DARK,
    group=2,
  )

  # Blade: the collision + visual box. Collides with the floor and the arm;
  # the fruit is collision-free, so slicing is decided geometrically.
  center_z = HILT_LENGTH + LENGTH / 2
  body.add_geom(
    name=BLADE_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(WIDTH / 2, THICKNESS / 2, LENGTH / 2),
    pos=(0, 0, center_z),
    mass=BLADE_MASS,
    rgba=_STEEL,
    group=2,
  )

  # Edge marker: visual only (no contacts, no mass), so the edge is easy to see.
  body.add_geom(
    name="blade_edge_marker",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.002, THICKNESS / 2 + 0.0005, LENGTH / 2),
    pos=(WIDTH / 2, 0, center_z),
    rgba=_EDGE,
    group=2,
    contype=0,
    conaffinity=0,
    mass=0,
  )

  site_kwargs = dict(size=(0.006, 0, 0), group=2)
  body.add_site(name=BLADE_CENTER_SITE, pos=(0, 0, center_z), rgba=(0, 1, 0, 1), **site_kwargs)
  body.add_site(name=EDGE_BASE_SITE, pos=(WIDTH / 2, 0, HILT_LENGTH), rgba=(1, 1, 0, 1), **site_kwargs)
  body.add_site(name=EDGE_TIP_SITE, pos=(WIDTH / 2, 0, HILT_LENGTH + LENGTH), rgba=(1, 1, 0, 1), **site_kwargs)
