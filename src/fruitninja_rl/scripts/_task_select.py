"""Shared by train.py and play.py: map ``--fruit-mode`` to a registered task."""

import sys

from fruitninja_rl.tasks import MULTI_FRUIT_TASK, SINGLE_FRUIT_TASK

_TASKS = {"multi": MULTI_FRUIT_TASK, "single": SINGLE_FRUIT_TASK}


def select_task() -> None:
  """Pop ``--fruit-mode {multi,single}`` (default multi) from sys.argv and put the
  matching task id first, unless the user already named a task."""
  mode = "multi"
  argv = sys.argv[1:]
  for i, arg in enumerate(argv):
    if arg == "--fruit-mode" and i + 1 < len(argv):
      mode = argv[i + 1]
      del argv[i : i + 2]
      break
    if arg.startswith("--fruit-mode="):
      mode = arg.split("=", 1)[1]
      del argv[i]
      break
  if mode not in _TASKS:
    sys.exit(f"--fruit-mode must be one of {sorted(_TASKS)}, got {mode!r}")
  sys.argv[1:] = argv
  if not argv or argv[0].startswith("-"):
    sys.argv.insert(1, _TASKS[mode])
