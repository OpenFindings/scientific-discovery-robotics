"""Run an openpi script with the pilot configs registered.

openpi's scripts look configs up in a module-level registry, so they are
run here in-process after openpi_ext.register(). Run from inside the openpi
checkout's environment (`uv run`), with OPENPI_ROOT pointing at it:

  uv run python $PILOT/run_openpi.py describe pilot_R
  uv run python $PILOT/run_openpi.py norm_stats --config-name pilot_R
  uv run python $PILOT/run_openpi.py train pilot_R --exp-name s41 --seed 41 --overwrite
"""
import json
import os
import pathlib
import runpy
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import openpi_ext  # noqa: E402

SCRIPTS = {"train": "train.py", "norm_stats": "compute_norm_stats.py"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in (*SCRIPTS, "describe"):
        sys.exit(f"usage: run_openpi.py {{{'|'.join([*SCRIPTS, 'describe'])}}} [args...]")
    names = openpi_ext.register()
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "describe":
        for n in rest or names:
            print(json.dumps(openpi_ext.describe(n)))
        return
    root = pathlib.Path(os.environ.get("OPENPI_ROOT", ".")).resolve()
    script = root / "scripts" / SCRIPTS[cmd]
    if not script.exists():
        sys.exit(f"{script} not found; set OPENPI_ROOT to the openpi checkout")
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
