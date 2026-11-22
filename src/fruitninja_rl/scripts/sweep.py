"""Hyperparameter sweep: one training run per grid point, then rank the runs.

Each grid point gets an integer index, so any job runner can launch one run
per index (a Slurm job array, a cluster queue, or a plain for-loop):

  uv run python -m fruitninja_rl.scripts.sweep --list            # show the grid
  uv run python -m fruitninja_rl.scripts.sweep --index 3 --dry-run
  uv run python -m fruitninja_rl.scripts.sweep --index 3         # train run 3
  SLURM_ARRAY_TASK_ID=3 uv run python -m fruitninja_rl.scripts.sweep   # same
  uv run python -m fruitninja_rl.scripts.sweep --collect         # rank results

Anything after ``--`` is passed through to train, e.g.
  ... sweep --index 3 -- --env.scene.num-envs 2048 --agent.max-iterations 500

Edit GRID to choose what to tune. Keys are train CLI flags (see
``uv run python -m fruitninja_rl.scripts.train --help``). A reward ``params``
entry is only overridable if it is listed in that term's ``params`` dict in
slicing_env_cfg.py.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

SWEEP_NAME = "slice_v1"
GRID: dict[str, list] = {
  "--env.rewards.staged-slice-reward.weight": [0.5, 1.0, 2.0],
  "--env.rewards.staged-slice-reward.params.reaching-std": [0.2, 0.4],
  "--env.rewards.staged-slice-reward.params.miss-std": [0.1, 0.3],
  "--agent.algorithm.entropy-coef": [0.002, 0.005],
}
# Applied to every run.
COMMON_ARGS = [
  "--env.scene.num-envs", "4096",
  "--agent.max-iterations", "1500",
]
LOG_ROOT = Path("logs/rsl_rl")
EXPERIMENT = "iiwa_slice"  # rl_cfg.py experiment_name
METRIC = "Episode_Metrics/slice_success"


def grid_points() -> list[dict[str, object]]:
  keys = list(GRID)
  return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*GRID.values())]


def run_name(index: int) -> str:
  return f"{SWEEP_NAME}_{index:03d}"


def train_command(index: int, extra: list[str]) -> list[str]:
  point = grid_points()[index]
  cmd = [sys.executable, "-m", "fruitninja_rl.scripts.train", *COMMON_ARGS]
  for flag, value in point.items():
    cmd += [flag, str(value)]
  cmd += [
    "--agent.run-name", run_name(index),
    "--agent.seed", str(index),
    "--env.seed", str(index),
    "--log-root", str(LOG_ROOT),
    *extra,
  ]
  return cmd


def final_score(run_dir: Path, tail_frac: float = 0.1) -> tuple[float, int] | None:
  """Mean of METRIC over the last ``tail_frac`` of logged iterations."""
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  acc.Reload()
  if METRIC not in acc.Tags().get("scalars", []):
    return None
  values = [e.value for e in acc.Scalars(METRIC)]
  tail = values[-max(1, int(len(values) * tail_frac)) :]
  return sum(tail) / len(tail), len(values)


def collect(log_root: Path) -> None:
  exp_dir = log_root / EXPERIMENT
  points = grid_points()
  rows = []
  for index in range(len(points)):
    # Run dirs are "<date>_<run_name>"; take the newest if it was rerun.
    dirs = sorted(exp_dir.glob(f"*_{run_name(index)}"))
    result = final_score(dirs[-1]) if dirs else None
    rows.append((index, result, dirs[-1].name if dirs else "-"))

  rows.sort(key=lambda r: -1.0 if r[1] is None else r[1][0], reverse=True)
  short = [k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1] for k in GRID]
  print(f"{'idx':>4} {'success':>8} {'iters':>6}  " + "  ".join(short))
  for index, result, _ in rows:
    score, iters = (f"{result[0]:.3f}", str(result[1])) if result else ("-", "-")
    vals = "  ".join(f"{str(v):>{len(s)}}" for v, s in zip(points[index].values(), short))
    print(f"{index:>4} {score:>8} {iters:>6}  {vals}")
  print(f"\n{METRIC}, mean over the last 10% of iterations; '-' = not run or no data.")


def main() -> None:
  argv = sys.argv[1:]
  extra = argv[argv.index("--") + 1 :] if "--" in argv else []
  own = argv[: argv.index("--")] if "--" in argv else argv

  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--index", type=int, help="grid point (default: $SLURM_ARRAY_TASK_ID)")
  parser.add_argument("--list", action="store_true", help="print the grid and exit")
  parser.add_argument("--dry-run", action="store_true", help="print the command only")
  parser.add_argument("--collect", action="store_true", help="rank finished runs")
  parser.add_argument("--log-root", type=Path, default=LOG_ROOT, help="for --collect")
  args = parser.parse_args(own)

  points = grid_points()
  if args.list:
    for i, p in enumerate(points):
      print(i, p)
    print(f"{len(points)} runs -> use a job array of 0-{len(points) - 1}")
    return
  if args.collect:
    collect(args.log_root)
    return

  index = args.index
  if index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
  if index is None or not 0 <= index < len(points):
    parser.error(f"need --index in [0, {len(points) - 1}] (or SLURM_ARRAY_TASK_ID)")

  cmd = train_command(index, extra)
  print("[sweep]", " ".join(cmd), flush=True)
  if not args.dry_run:
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
  main()
