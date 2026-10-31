"""MDP terms for the slicing task.

Re-exports mjlab's built-in terms (joint_pos_rel, action_rate_l2, time_out, ...)
alongside the task-specific ones, so the env config can use ``mdp.<term>`` for
everything.
"""

from mjlab.envs.mdp import *  # noqa: F403

from .metrics import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
