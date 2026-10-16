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
| `openpi_ext.py` | openpi env | Registers `pilot_N`, `pilot_R`, `pilot_C` and `pilot_base_probe` into openpi's config registry. `pilot_C` reads `local/libero_relstate`. Each is stock `pi05_libero` with the Blackout-vs-Freeze sim recipe: 10k updates, batch 32, 10k linear warmup to 5e-5, EMA 0.999. Only `discrete_state_input` differs between arms. |
| `relative_state.py` | both envs (numpy only, Py 3.8+) | Arm C encoding: `pos − pos₀`, `axis_angle(R₀ᵀ R)`, gripper absolute. One function serves both the training conversion and the eval client, so the two cannot drift apart. |
| `convert_relative_state.py` | openpi env | Copies `physical-intelligence/libero` (LeRobot v2.0) to `$HF_LEROBOT_HOME/local/libero_relstate`, rewriting only the `state` column. Image bytes, actions, indices and HF schema metadata are copied unchanged. Before writing each episode it checks frame 0 = 0 and the absolute state round-trips. It updates `meta/stats.json` and writes a manifest of per-episode anchors. Needs ~30 GB. |
| `check_anchors.py` | numpy + pyarrow | Arm C gate. Compares each unperturbed eval episode's logged `start_state` with the frame-0 states of that task's training demos. Reports the **systematic** mean offset (mm, degrees, and C state-token bins), **coverage** (share of eval starts inside the training-anchor cloud), and **pairing** (the same instance starts identically across arms and seeds). Reads anchors from a local dataset or straight from Hugging Face via HTTP range requests, so it runs in Stage 0 before the 30 GB conversion. |
| `run_openpi.py` | openpi env | Runs openpi's `train.py` / `compute_norm_stats.py` with those configs registered. `describe` prints the settings that must match the design. |
| `serve.py` | openpi env | Websocket policy server. On request, it returns the **decoded prompt** the model actually saw, so a train/serve flag mismatch is logged. `--norm-stats-from` supplies LIBERO stats for `pi05_base`. |
| `eval_libero.py` | LIBERO client venv (Py 3.8) | Runs paired instances (suite × task × init state) under each condition and writes one JSON line per episode: success, steps, prompt, per-query state, perturbation and fault timing. Supports `id`, `scene_blackout`, `wrist_blackout_after_close`, `robot_init:<m>` and `layout:<m>` (Stage 1). `--state-mode relative` (arm C) sends the state relative to the first policy query. |
| `layout_perturb.py` | LIBERO client venv (Py 3.8) | Stage 1 K-layout: moves the task's first movable object of interest exactly `m` metres in a random direction on the LIBERO init state. It checks the placement over a 10-step settle (no object pushed more than 10 mm, no drop over 30 mm, no tilt over 15°, task not already solved), retries up to 8 deterministic directions, and restores the validated state exactly. Instances with no movable target or no valid placement are logged as `skipped`. |
| `analyze_stage0.py` | any Python with numpy | Reports success tables and checks 1, 2, 3 and 5, plus a prompt-flag check. Gives the seed SD, the Stage 1 seed recommendation, and the share of R's state tokens that are saturated or out of range. |
| `stage0.sh` | shell | Orchestrates the steps: `norm_stats train eval anchors reference probe analyze` (plus `convert` for arm C). |
| `test_stage0_offline.py` | any Python with numpy (+ pyarrow for the converter test) | 21 offline tests. |

## Choices worth knowing

- **Robot initial state** reproduces LIBERO-Plus exactly in kind. Its 500 released offsets have L2 norms of exactly 0.1, 0.2, 0.3, 0.4 and 0.5 rad over the 7 arm joints, in random directions. Here the offset is `m · u` with `u` uniform on the 7-D unit sphere, clipped to joint limits. It is applied after the LIBERO init state and before the 10 settle steps. The OSC null-space target moves with it (`update_initial_joints`), as it does when LIBERO-Plus swaps the robot's `init_qpos`.
- **Pairing.** Perturbation directions and fault delays are seeded only by (suite, task, init state, condition). Every arm and seed therefore sees identical instances.
- **Faults** follow Blackout vs Freeze App. A.2. Blackout zeroes the preprocessed image. The after-closure wrist fault triggers when finger width stays below 0.020 m for 3 policy steps, then waits U{0..10} steps.
- **The checkpoint evaluated is `…/9999`.** openpi saves the final step as `num_train_steps − 1`, and the saved params are the EMA weights.
- **Check 5 was relaxed** from "N within 10–90%" to "N and R not both at floor or both at ceiling". Blackout vs Freeze's no-state π0.5 sits at 9.1% under scene blackout, so the original rule would fail by design.

## Simulator pitfalls found by running LIBERO here (CPU, MuJoCo)

- **Pin `mujoco>=3.2.3`.** This is openpi's own pin in `examples/libero/requirements.txt`. With mujoco 3.1.6, every hard `env.reset()` leaks about 50 file descriptors (the Panda visual meshes), measured over 20 resets: +50/reset on 3.1.6, +0 on 3.2.3. That exhausts the descriptor limit after a few hundred episodes.
- **LIBERO's `ControlEnv.reset()` can hang silently.** It retries inside `while … try … finally: continue`, and the `finally: continue` swallows every exception. Any reset failure, such as "too many open files", becomes a silent infinite loop. Four probe processes stalled this way, all at the same reset count.

  `eval_libero.py` now resets through `_safe_reset()`, which retries only robosuite's `RandomizationError` (up to 20 times) and raises everything else. Each episode row also logs `open_fds`, so a leak is visible in the data.
- **Objects fall about 7 cm during the 10 settle steps** from LIBERO init states, even with no perturbation. Layout placements are therefore validated against an unperturbed settle, not against the placement pose (see `layout_perturb.py`).

## K-layout notes (Stage 1)

- **Why not LIBERO-Plus's layout files.** Measured over the four suites, its `levelK_sampleN` BDDL files shift *every* object region:
  - maximum shift 30·K mm, median about 8·K mm;
  - 6% of regions also get a yaw range;
  - the instances come with regenerated init states.

  Those instances cannot be paired with the LIBERO init states the other conditions use. This pilot instead moves one object on the unchanged init state, at exactly 3 cm or 6 cm (LIBERO-Plus's level-1 and level-2 maxima), with the direction seeded by instance.
- **Which object.** The first entry of the BDDL `obj_of_interest` that is movable. In spatial, object and most goal tasks this is the grasped object. In two LIBERO-Long tasks it is a destination plate or the first of two moka pots. The moved object is logged per episode.

  "Open the middle drawer" and "turn on the stove" have only fixtures of interest and are skipped: 20 of 400 instances per magnitude.
- **Not yet measured in MuJoCo.** The validity thresholds (10 mm, 30 mm, 15°) are set for real contact physics but were only exercised against a toy simulator. Before Stage 1, run `layout:0.03,layout:0.06` with one checkpoint and check two things: the share of instances needing retries or skipped, and that the moved object is visibly displaced in the logged videos (`--video-dir`).

## Arm C notes

- **Anchors.** The training anchor is the dataset's frame 0. The eval anchor is the state at the first policy query, after the 10 settle steps. `check_anchors.py` (the `anchors` step) tests whether they line up.

  A sample of 212 real training episodes (every 8th, read over HTTP) showed:
  - **Position is in the world frame, not the robot base frame.** Anchors form two clusters about 0.9 m apart, one per scene layout. Compare within task only.
    - Arm R's absolute state therefore also identifies the scene layout.
  - **Frame-0 positions vary within a task** by up to 17–28 mm (about 10 mm per axis typically, mostly in a plane).
  - **Orientation and gripper barely vary**: orientation by less than 0.5°, and the gripper is always fully open.

  So the pre-set rule tests a *systematic* offset rather than equality:
  - mean offset ≤ 5 mm per axis and ≤ 1°;
  - ≥ 95% of eval starts inside the pooled within-task 99th-percentile cloud (+2 mm, +0.5°);
  - every task matched;
  - same-instance starts identical across arms and seeds to 0.1 mm.

  Validated on the real anchors by holding out half the sampled episodes as pseudo-eval starts. The mean offset was 0.5 mm and passed; an injected 10 mm shift failed (3.5 C-state bins).

  Coverage in that test was only 91%. The half-sample had just 2–3 demos per task, so the task means were noisy. The real check uses all ~42 demos per task.
- **Relative orientation also fixes an encoding wart.** LIBERO's absolute axis-angle sits around 2.94–3.24 rad, straddling π: the gripper points down. That representation is not unique there. `R₀ᵀ R` is near identity, so the relative encoding is small and continuous.

  The inverse (used only for verification) handles near-π exactly via `kkᵀ = ((R+Rᵀ)/2 − cos θ I)/(1 − cos θ)`. A first-order version left 1e-4 errors on real LIBERO episodes, which the converter's round-trip check caught.
- **Tested on real data.** I converted 3 real LIBERO episodes (0, 1, 500) with `--max-episodes`. Frame 0 came out exactly zero, the round-trip error was ≤ 2e-8, and every non-state column and the schema metadata were identical to the source.

## Run

```bash
OPENPI_ROOT=~/openpi bash pilot/stage0/stage0.sh            # all steps
OPENPI_ROOT=~/openpi bash pilot/stage0/stage0.sh eval analyze
OPENPI_ROOT=~/openpi ARMS="C" bash pilot/stage0/stage0.sh convert norm_stats train eval analyze   # arm C
SEEDS="41 42" ARMS="N R" CONDITIONS="id,scene_blackout,robot_init:0.2" ...   # overrides
```

Outputs go to `pilot/results/stage0/`:
- `<arm>_s<seed>.jsonl`, one line per episode;
- `summary.json`;
- `configs.jsonl`;
- `logs/`.
