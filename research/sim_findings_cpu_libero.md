# What a CPU LIBERO study says about the pilot

> **Update 2026-10-17.** Superseded in part by `research/paper_draft.md`. That draft adds 3 seeds for the blind and ego policies and real-image (pixel CNN) policies. Two results here **changed with pixels**:
> - under the robot-start shift, raw state is neutral (R−N +4, CI includes 0), not harmful;
> - relative state helps (C−R +8).
>
> The ego-oracle harm replicates over 3 seeds. It reflects the no-state arm being blind to the arm's configuration (paper §4.3).

*2026-10-16. All runs in real LIBERO on CPU (MuJoCo 3.2.3, robosuite 1.4.1, no GPU), with the pilot's own perturbation code. Scripts and per-episode results: `pilot/sim/`. Context: `research/pilot_3x2_liberoplus.md`, `evidence_map/README.md`.*

π0.5 cannot be trained here, since it needs a GPU with more than 70 GB. What *can* run is the simulator, the pilot's perturbations, and small policies whose inputs are fully controlled. Four experiments used them:

1. **E1. Policy-free probe.** All 400 pilot instances (4 suites × 10 tasks × 10 init states) under ID, robot-init 0.1 / 0.2 / 0.3, and layout 3 cm / 6 cm: 2,400 episode starts.
2. **E2. Blind policies (no perception).** Task plus one input, predicting 10-step action chunks; trained on all 1,693 openpi LIBERO demos; 959 paired instances, about 2,900 episodes. The input is one of:
   - **T:** the time step (open-loop replay);
   - **A:** absolute state (arm R's encoding);
   - **Rl:** episode-relative state (arm C's).
3. **E3. Allocentric oracle.** Task plus the world positions of the two objects of interest ("perfect scene camera"), with no state (O), absolute state (O+A) or relative state (O+Rl). LIBERO-Spatial and LIBERO-Object; trained on the original LIBERO demos, whose object poses I rebuilt by replaying their saved simulator states; 579 paired instances.
4. **E4. Ego-centric oracle.** As E3, but the object positions are given relative to the gripper ("perfect wrist camera"): E, E+A, E+Rl. This is the closest analogue of the pilot's arms N, R and C. 579 paired instances.

In every experiment, every policy runs on identical instances and perturbations. Differences are paired, with 95% task-cluster bootstrap CIs. Success rates are in %.

## Conclusions

1. **LIBERO in-distribution success says almost nothing about whether a policy uses state well.**
   - A blind MLP with absolute state and no images solves **71.5%** of ID instances; on LIBERO-Object it solves 94%.
   - Open-loop replay of the time-indexed mean trajectory solves 52.5%.
   - Raw state explains about 85% of the action-chunk variance in the demos.

   This supports the evidence map's reading that the in-distribution residual is small and uninformative. It also explains why ID differences between with- and without-state arms are a few points at most.
2. **Robot-initial-state shift: state hurts in every regime with perception, and the frame correction does not fix it.**
   - **Ego oracle at 0.1 rad:** E 81 vs E+A 61 vs E+Rl 59. R−N = −20 [−35, −5]; C−N = −22 [−32, −12]; C−R = −2 [−14, +9].
   - **At 0.2 rad:** R−N = −18 [−30, −6]; C−N = −23 [−35, −11].
   - **Blind regime:** absolute state does worse than open-loop replay (A−T = −18.5 [−26, −11] at 0.1). The memorized state→action map extrapolates wrongly when the arm starts somewhere new.

   This is the H6 pattern, not H7. Hurt under robot-config shift is robust across regimes; whether relative state helps depends on what else the policy perceives:

   | Regime | Relative (Rl) − absolute (A) at robot-init 0.1 |
   |---|---|
   | blind | **+20** [+11.5, +28.5] |
   | allocentric oracle | **−34** [−45, −23] |
   | ego oracle | −2 [−14, +9] |

   With world-frame object information, relative state throws away the absolute arm position needed to reach the object. With ego-centric information it adds nothing. Only a policy that cannot localize objects, and so is replaying, benefits from it.
3. **Layout shift behaves differently from robot-init shift; "configuration shift" should not be pooled.** With good perception (ego oracle):
   - at 3 cm, state is neutral (R−N = +5.5, CI includes 0);
   - at 6 cm, state helps: R−N = +11 [−1, +23]; C−N = **+18** [+6, +30].

   Under the evidence map's grouping, published harm under object-position shifts (2509.18644, 2602.12032; real VLAs) and this simulated benefit can both hold only if real policies' visual grounding is imperfect. Raw state would then be harmful through **vision suppression**, not intrinsically.

   That makes the pilot's K-layout contrast a direct measurement of vision suppression in π0.5. The K-robot contrast measures something else: a state input that is itself out of the training trajectory.
4. **The pilot's single decisive cell is less decisive than designed.** The design treated CK (relative state vs none, under configuration shift) as the H6-vs-H7 decider. The simulations predict its sign depends on the sub-condition:
   - robot-init: CK hurts, as RK does;
   - layout: CK helps if π0.5 grounds objects well.

   Pooling the two could cancel them out. Analyze K-robot and K-layout separately (changed in the design, below).
5. **The camera-fault analogue was uninformative.** Removing all object information (oracle "blackout") gave 0% for every policy, so state could not rescue anything. The pilot's single-camera faults, as in Blackout vs Freeze, remain the only test of the F half; nothing here predicts it.
6. **The pilot infrastructure works in the real simulator, after four fixes.** These are listed under Infrastructure below.

## E1. Policy-free probe (all 400 instances × 6 conditions)

| | |
|---|---|
| **Arm C anchor gate** (`check_anchors.py`, all 1,693 training anchors) | **Passes.** Mean eval-start offset (−4.5, −0.2, +0.5) mm, i.e. 1.2 C-state bins in x, close to the 5 mm limit. Coverage 99.25%. Same-instance starts identical across runs (0.0 mm). |
| **Robot-init EEF displacement** (median / p90) | 0.1 rad: 35 / 56 mm. 0.2 rad: 70 / 105 mm. 0.3 rad: 103 / 159 mm. No joint-limit clipping. |
| **Share of raw start states outside the training 1–99% range** | 0% / 2.5% / 14% at 0.1 / 0.2 / 0.3. At the planned magnitudes, arm R's state tokens are *in range but wrong*; harm cannot come from token saturation. |
| **Layout validity** (valid / skipped) | Object and Long: 100% valid. Goal: 80% (its 2 fixture-only tasks are skipped by design). Spatial: 86% at 3 cm, 77% at 6 cm (crowded scenes). Mean tries 1.0–1.65. No instance pre-solved. |

## E2–E4. Results

**Blind (E2; 4 suites):**

| Condition | n | T | A | Rl | Rl − A | A − T |
|---|---|---|---|---|---|---|
| id | 200 | 52.5 | 71.5 | 65.5 | −6.0 [−13.5, +1.5] | +19.0 [+6.5, +31.5] |
| robot_init 0.1 | 200 | 30.0 | 11.5 | 31.5 | **+20.0** [+11.5, +28.5] | **−18.5** [−26, −11] |
| robot_init 0.2 | 200 | 16.0 | 8.0 | 11.0 | +3.0 [−2, +8] | −8.0 [−13.5, −2.5] |
| layout 3 cm | 181 | 30.4 | 43.6 | 38.7 | −5.0 [−12.5, +2.7] | +13.3 [+3.8, +22.7] |
| layout 6 cm | 178 | 3.9 | 3.4 | 6.7 | +3.4 [−0.6, +8.0] | −0.6 [−4.5, +3.4] |

**Allocentric oracle (E3; Spatial + Object).** O (objects, no state) cannot act, at 5% ID, because nothing tells it where the arm is. So it is not a fair no-state arm:

| Condition | n | O | O+A | O+Rl | Rl − A |
|---|---|---|---|---|---|
| id | 100 | 5.0 | 77.0 | 68.0 | −9.0 [−21, +2] |
| robot_init 0.1 | 100 | 8.0 | 56.0 | 22.0 | **−34.0** [−45, −23] |
| robot_init 0.2 | 100 | 3.0 | 24.0 | 12.0 | −12.0 [−25, 0] |
| layout 3 cm | 91 | 0.0 | 72.5 | 51.6 | **−20.9** [−35.8, −5.0] |
| layout 6 cm | 88 | 1.1 | 47.7 | 22.7 | **−25.0** [−40.7, −8.5] |
| blackout | 100 | 0 | 0 | 0 | — |

**Ego-centric oracle (E4; Spatial + Object): the pilot's N / R / C analogue:**

| Condition | n | E (N) | E+A (R) | E+Rl (C) | R − N | C − N | C − R |
|---|---|---|---|---|---|---|---|
| id | 100 | 93.0 | 90.0 | 86.0 | −3.0 [−10, +4] | **−7.0** [−13, −1] | −4.0 [−14, +6] |
| robot_init 0.1 | 100 | 81.0 | 61.0 | 59.0 | **−20.0** [−35, −5] | **−22.0** [−32, −12] | −2.0 [−14, +9] |
| robot_init 0.2 | 100 | 57.0 | 39.0 | 34.0 | **−18.0** [−30, −6] | **−23.0** [−35, −11] | −5.0 [−17, +6] |
| layout 3 cm | 91 | 72.5 | 78.0 | 71.4 | +5.5 [−5.7, +16] | −1.1 [−13, +10] | −6.6 [−21, +7.5] |
| layout 6 cm | 88 | 30.7 | 42.0 | 48.9 | +11.4 [−1.2, +23] | **+18.2** [+5.6, +30.3] | +6.8 [−3.5, +17.4] |
| blackout | 100 | 0 | 0 | 0 | — | — | — |

**Training losses** (normalized MSE on 10-step chunks):

| Experiment | Without state | + absolute state | + relative state |
|---|---|---|---|
| Blind | T 0.44 | A 0.157 | Rl 0.142 |
| Allocentric oracle | O 0.47 | O+A 0.078 | O+Rl 0.064 |
| Ego oracle | E 0.169 | E+A 0.066 | E+Rl 0.067 |

Even with perfect ego-centric object information, state cuts the imitation loss by 2.5×. That is the optimization pull behind the shortcut hypothesis. In this idealized setting it did *not* stop the policy from using object information under layout shift.

## What changes in the pilot

- **Analyze K-robot and K-layout separately**, each with its own pre-registered sign predictions, and do not pool them. Predictions:
  - **K-robot:** RK hurt and CK hurt. Simulation predicts the relative frame does not remove the harm when the policy perceives the scene.
  - **K-layout:** RK and CK not hurt, or help, if π0.5 grounds objects visually. A harm here measures **vision suppression** by state.

  Under this reading:
  - H7 is supported only if CK beats RK under robot-init. The ego analogue says it will not (−2).
  - H6 is supported if both hurt under robot-init.
- **Primary magnitudes.**
  - **Layout: 6 cm.** Blind replay solves only 3–7% there, versus 30–44% at 3 cm, so 6 cm actually requires perception. 3 cm becomes secondary.
  - **Robot-init: keep 0.1 and 0.2.** 0.3 adds little beyond floor effects.
- **Report a blind floor per condition.** Run the blind-A and blind-T policies (`pilot/sim/blind_*`) on the same instances. A π0.5 arm that does no better than blind replay under a shift is not using vision there.
- **The arm C anchor gate has passed.** No change needed before training C, but the x-offset (−4.5 mm) is close to the limit; recheck it with the real eval logs.

## Infrastructure: bugs found by running LIBERO (fixed in `pilot/stage0/`)

1. **mujoco 3.1.6 leaks about 50 file descriptors per hard reset; 3.2.3 (openpi's pin) leaks none.** Measured: +50 per reset vs +0 over 20 resets.
2. **LIBERO's `ControlEnv.reset()` swallows every exception in an infinite retry loop** (`finally: continue`). Combined with the leak, four probe processes hung silently. Now handled by `_safe_reset()` plus per-episode `open_fds` logging.
3. **Objects fall about 7 cm during the 10 settle steps** from LIBERO init states. Layout validation now compares against an unperturbed settle; tolerances were tightened after 10–13° rim-leaning placements were accepted.
4. **Restoring the saved MuJoCo state after validation left the robot 1–7 mm (up to 13 mm) off its paired ID start.** The fix re-enters the placement through a fresh reset; the offset is now 0.0 mm. E2's layout rows were run before this fix and carry that small offset. E3, E4 and the pilot code do not.

## Caveats

- **Small MLPs, not π0.5.** Oracle inputs are perfect object positions, not pixels, and there is one training seed per policy. The results bound *mechanisms*; they do not predict π0.5's numbers.
- **Coverage is partial.** The oracles cover 2 of 4 suites (Spatial, Object) and 5 init states per task. The blind policies cover all 4 suites.
- **The oracle "no state" arms differ from the pilot's.**
  - Allocentric O has no arm information at all.
  - Ego E carries arm-object geometry, as a wrist camera does, but nothing about the arm's own configuration.

  The real N arm sees the arm in its images, so it sits between these two.
- **The camera-fault analogue was too harsh to be informative.**
- **These are simulation analogues, not literature results.** They are kept out of the evidence map's `results.jsonl`.
