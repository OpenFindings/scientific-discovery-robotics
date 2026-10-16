# Stage 0 scripts (openpi)

Stage 0 of `research/pilot_3x2_liberoplus.md` (§8):
- trains arms **N** (no state) and **R** (raw state) with seeds 41 and 42;
- evaluates them under ID, scene blackout and robot initial state (magnitude 0.2);
- serves two reference checkpoints:
  - the released `pi05_libero` (openpi's own no-state model);
  - a `pi05_base` zero-shot LIBERO probe;
- applies the exit criteria.

Written against openpi @ `215abfb` (2026-08-24) and LIBERO-plus @ `4976dc3`. **None of these scripts has been run on a GPU or in LIBERO yet.** `test_stage0_offline.py` checks the client and analysis logic with fakes.

| File | Runs in | What it does |
|---|---|---|
| `openpi_ext.py` | openpi env | Registers `pilot_N`, `pilot_R` and `pilot_base_probe` into openpi's config registry. Each is stock `pi05_libero` with the Blackout-vs-Freeze sim recipe: 10k updates, batch 32, 10k linear warmup to 5e-5, EMA 0.999. Only `discrete_state_input` differs between arms. |
| `run_openpi.py` | openpi env | Runs openpi's `train.py` / `compute_norm_stats.py` with those configs registered. `describe` prints the settings that must match the design. |
| `serve.py` | openpi env | Websocket policy server. On request, it returns the **decoded prompt** the model actually saw, so a train/serve flag mismatch is logged. `--norm-stats-from` supplies LIBERO stats for `pi05_base`. |
| `eval_libero.py` | LIBERO client venv (Py 3.8) | Runs paired instances (suite × task × init state) under each condition and writes one JSON line per episode: success, steps, prompt, per-query state, perturbation and fault timing. Supports `id`, `scene_blackout`, `wrist_blackout_after_close` and `robot_init:<m>`. |
| `analyze_stage0.py` | any Python with numpy | Reports success tables and checks 1, 2, 3 and 5, plus a prompt-flag check. Gives the seed SD, the Stage 1 seed recommendation, and the share of R's state tokens that are saturated or out of range. |
| `stage0.sh` | shell | Orchestrates the steps: `norm_stats train eval reference probe analyze`. |
| `test_stage0_offline.py` | any Python with numpy | 9 offline tests. |

## Choices worth knowing

- **Robot initial state** reproduces LIBERO-Plus exactly in kind. Its 500 released offsets have L2 norms of exactly 0.1, 0.2, 0.3, 0.4 and 0.5 rad over the 7 arm joints, in random directions. Here the offset is `m · u` with `u` uniform on the 7-D unit sphere, clipped to joint limits. It is applied after the LIBERO init state and before the 10 settle steps. The OSC null-space target moves with it (`update_initial_joints`), as it does when LIBERO-Plus swaps the robot's `init_qpos`.
- **Pairing.** Perturbation directions and fault delays are seeded only by (suite, task, init state, condition). Every arm and seed therefore sees identical instances.
- **Faults** follow Blackout vs Freeze App. A.2. Blackout zeroes the preprocessed image. The after-closure wrist fault triggers when finger width stays below 0.020 m for 3 policy steps, then waits U{0..10} steps.
- **The checkpoint evaluated is `…/9999`.** openpi saves the final step as `num_train_steps − 1`, and the saved params are the EMA weights.
- **Check 5 was relaxed** from "N within 10–90%" to "N and R not both at floor or both at ceiling". Blackout vs Freeze's no-state π0.5 sits at 9.1% under scene blackout, so the original rule would fail by design.

## Run

```bash
OPENPI_ROOT=~/openpi bash pilot/stage0/stage0.sh            # all steps
OPENPI_ROOT=~/openpi bash pilot/stage0/stage0.sh eval analyze
SEEDS="41 42" ARMS="N R" CONDITIONS="id,scene_blackout,robot_init:0.2" ...   # overrides
```

Outputs go to `pilot/results/stage0/`:
- `<arm>_s<seed>.jsonl`, one line per episode;
- `summary.json`;
- `configs.jsonl`;
- `logs/`.
