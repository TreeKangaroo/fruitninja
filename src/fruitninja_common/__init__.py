"""Code shared by the RL and trajectory-optimization sides: robot, fruit, cut condition."""

# Import mjlab before anything else in this package. On import, mjlab
# auto-loads registered task packages (our fruitninja_rl.tasks), which in turn
# import fruitninja_common.*. Initializing mjlab first ensures that happens
# before any of our modules is half-loaded, avoiding a circular import when a
# script (e.g. trajopt) imports fruitninja_common directly.
import mjlab  # noqa: F401
