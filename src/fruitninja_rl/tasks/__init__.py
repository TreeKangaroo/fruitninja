"""Registers the fruit-slicing tasks (multi-fruit and single-fruit) with mjlab.

mjlab imports this package through the ``mjlab.tasks`` entry point declared in
pyproject.toml, which makes the tasks available to ``uv run train``/``play``.
"""

from mjlab.tasks.registry import register_mjlab_task

from .rl_cfg import make_slicing_ppo_cfg
from .slicing_env_cfg import make_slicing_env_cfg

MULTI_FRUIT_TASK = "Fruitninja-Slice-Iiwa"
SINGLE_FRUIT_TASK = "Fruitninja-Slice-Iiwa-Single"

# Both tasks mix every trajectory in fruit.TRAJECTORIES and have the same
# observation/action spaces, so checkpoints transfer between the two. They do not
# transfer from straight-drop-only training: that observation normalizer has zero
# variance for the fruit's vx, so train afresh.
for task_id, multi in ((MULTI_FRUIT_TASK, True), (SINGLE_FRUIT_TASK, False)):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_slicing_env_cfg(multi_fruit=multi),
    play_env_cfg=make_slicing_env_cfg(multi_fruit=multi),
    rl_cfg=make_slicing_ppo_cfg(),
  )
