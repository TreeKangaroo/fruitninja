"""Registers the fruit-slicing tasks with mjlab.

mjlab imports this package through the ``mjlab.tasks`` entry point declared in
pyproject.toml, which makes the tasks available to ``uv run train``/``play``.
"""

from mjlab.tasks.registry import register_mjlab_task

from .rl_cfg import make_slicing_ppo_cfg
from .slicing_env_cfg import make_slicing_env_cfg

register_mjlab_task(
  task_id="Fruitninja-Slice-Iiwa",
  env_cfg=make_slicing_env_cfg(),
  play_env_cfg=make_slicing_env_cfg(),
  rl_cfg=make_slicing_ppo_cfg(),
)
