# CPU LIBERO simulation study

Write-up and conclusions: `research/sim_findings_cpu_libero.md`.

Everything runs on CPU in real LIBERO, with no GPU and no π0.5:
- MuJoCo 3.2.3, robosuite 1.4.1, Python 3.11;
- `libero_boot.py` sets up OSMesa rendering and allows numpy arrays in torch-safe loading of LIBERO init files.

The scripts were run from a scratch directory, and several have absolute paths in their `sys.path` lines and arguments; adjust them before rerunning. They import the pilot's own code from `pilot/stage0/`: `eval_libero`, `layout_perturb`, `relative_state` and `check_anchors`.

| Script | What it does |
|---|---|
| `fetch_train.py` | Fetches `state` and `actions` of all 1,693 `physical-intelligence/libero` episodes over HTTP range requests (no images, about 8 min). |
| `probe_starts.py`, `analyze_probe.py`, `anchor_from_npz.py` | **E1.** Policy-free probe of start states under ID / robot-init / layout; the arm C anchor gate. |
| `blind_train.py`, `blind_eval.py` | **E2.** Blind MLP policies (task + time / absolute / relative state), evaluated closed-loop. |
| `oracle_extract.py` | Rebuilds per-step object positions for the original LIBERO HDF5 demos by replaying their saved MuJoCo states. Reads only states and actions over HTTP. |
| `oracle_train.py`, `oracle_eval.py` | **E3.** Allocentric oracle: object world positions ± state. |
| `ego_train.py`, `ego_eval.py` | **E4.** Ego-centric oracle: object minus gripper position ± state. The pilot's N / R / C analogue. |
| `analyze_blind.py` | Paired success table with task-cluster bootstrap CIs. Usage: `python3 analyze_blind.py "<glob>" POLICY,POLICY,...` |
| `fd_leak.py`, `ctrl_test.py` | Reproduce two infrastructure bugs: the mujoco 3.1.6 file-descriptor leak, and the robot-start offset after layout validation. |
| `launch_*.sh` | The parallel launches as they were run. |

`results/` holds per-episode JSONL for every experiment, plus `summary.txt`, the training logs, `probe/anchor_check.json` and `probe/probe_summary.json`.
