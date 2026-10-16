# Pilot design: state encoding × shift type on LIBERO-Plus

*Step 5.3 of `research/area_selection_from_paper3_paper4.md`, redesigned from the evidence map (`evidence_map/README.md`, §5). Drafted 2026-10-15. Config: `pilot/configs/pilot.yaml`. Scripts: `pilot/scripts/`.*

## 0. What the pilot decides

The evidence map leaves two live explanations for why proprioceptive state sometimes helps and sometimes hurts manipulation policies:

- **H6, shift type.** State hurts when the robot or scene *configuration* leaves the training range. It helps when the *cameras* fail. Encoding does not matter.
- **H7, shift type × encoding.** Under configuration shift, the harm comes from *raw* (absolute-frame, naively fused) state. A corrected encoding removes it and turns it into a gain. Under camera faults, state helps.

Current planner posterior: H7 0.67, H6 0.28, H1 0.05. H7 was formed after seeing the data, and the evidence for it under shift comes from three papers.

The two hypotheses disagree on exactly one cell: **corrected state vs no state under configuration shift.** No paper also checks whether the correction keeps state's benefit under camera faults. That second question, call it **T (trade-off)**, is untested.

The pilot runs on one backbone with fixed data, cameras and recipe:

- three state encodings: none, raw, corrected
- two shift types: configuration shift, camera fault
- an in-distribution baseline

## 1. Why LIBERO-Plus, and how to live with LIBERO's known problems

**Why LIBERO-Plus.** It perturbs **robot initial joint states** (`qpos`, generator magnitude 0.1–0.5) and **target-object pose** (sub-dimension O2) inside the same LIBERO scenes that Blackout vs Freeze used for camera faults. So both shift types can be applied to *the same* task instances. That pairing is what makes the comparison clean.

It is also where the problem is most visible. In LIBERO-Plus Table 1, π0 (which uses state) drops to **6.0%** under robot-initial-state perturbation. OpenVLA-OFT drops to 37.2%.

**The plan (§5.6) warned against using LIBERO as the primary evidence.** Four mitigations follow.

1. **Do not use the filtered LIBERO-Plus test set for the decision.** The 10,030 released instances were kept only if OpenVLA-OFT, π0, π0-fast and UniVLA failed them often enough (LIBERO-Plus App. C). Some of those models use state, so the selection is entangled with the variable under test. Instead, use the **LIBERO-Plus generator** at fixed magnitudes on fixed LIBERO initial states. Report the official subsets only as a secondary comparison.
2. **Pair everything.** Each of the 400 instances (task × initial state) is run under every condition, by every arm and every seed.
3. **Report per-episode outcomes** and the provenance roots: backbone `pi0.5`, data `LIBERO`.
4. **Plan a non-LIBERO replication** before claiming generality (Stage 2, §8).

## 2. Factors

### 2.1 State encoding (3 arms)

| Arm | State input | Notes |
|---|---|---|
| **N**, none | omitted | openpi's stock `pi05_libero` config (`discrete_state_input=False`). The prompt is `Task: …` with no state string, and π0.5 has no other state pathway (§2.4). The released `pi05_libero` checkpoint is this arm. |
| **R**, raw | `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`, world frame (LIBERO's `robot0_eef_pos`; robot bases sit ~0.9 m apart across scene layouts) | `pi05_libero` with `discrete_state_input=True`. The quantile-normalized state is cut into 256 bins and written into the prompt as `State: b1 … b8;` (naive fusion). This is what Blackout vs Freeze trained. |
| **C**, corrected | episode-relative: `pos_t − pos_0`, `axis_angle(R_0ᵀ R_t)`, gripper absolute | As R, plus a repack transform that subtracts the episode's first state before `Normalize`. Norm stats are recomputed with `compute_norm_stats.py` on the transformed data. |

**Why C is episode-relative state and not regularization.**

- It is a pure input transform with no new hyperparameters.
- It has the most support among the H7 cells: 2605.13067, ZETA, and Mirage's frame alignment.
- It makes a sharp prediction for robot-initial-state shift: the state starts at zero for every episode, so it is in-distribution by construction.

A state-token-dropout arm (**D**, regularized) is pre-planned as an extension (§8). It probably *trades away* the camera-fault benefit, because it teaches the policy to ignore state. That makes it the natural second test of T.

**What C does *not* trivially fix.**

- Under target-object pose shift, a relative state still allows "replay the demonstrated displacement".
- Under robot-initial-state shift, the image shows the arm somewhere new while the state says "at start". Vision must reconcile the two.

Either could make C still hurt. That possibility is what makes CK a real test of H7.

### 2.2 Test conditions

All conditions run on the same 400 instances: 4 suites × 10 tasks × LIBERO init-states 0–9.

| Condition | Kind | Definition | Role |
|---|---|---|---|
| **ID** | none | Unperturbed. | Baseline and convergence check. |
| **K-robot** | configuration | LIBERO-Plus robot-init generator, magnitudes 0.1, 0.2, 0.3. | Primary (pooled into K). |
| **K-layout** | configuration | LIBERO-Plus O2, target-object pose, two levels. No distractor objects (O1), which would add a visual/semantic shift. | Primary (pooled into K). |
| **F-scene** | camera fault | Agent-view camera blacked out from the first policy step. The wrist camera and state are untouched. | Primary F. Blackout vs Freeze: π0.5 28.9% with state vs 9.1% without, so it is off the floor. |
| **F-wrist** | camera fault | Wrist camera blacked out after gripper closure (finger width < 0.020 m for 3 steps, then a uniform 0–10 step delay; Blackout vs Freeze App. A). | Secondary. It tests *closed-loop* use of state mid-task; episode-start wrist faults are at the floor (π0.5 13.1 vs 1.8). |
| LP-official | mixed | Released LIBERO-Plus robot / layout / camera subsets. | Secondary, for comparability only. |

**Why not freezing.** Frozen-frame faults are near 0% in every arm (π0.5 5.8 vs 0.2), so they cannot separate the hypotheses.

**Why not camera-viewpoint shift.** It is a visual shift that does *not* remove information. H6 and H7 make no clean prediction for it, so it is kept out of the decision rule. It is reported in LP-official.

### 2.3 Held fixed

- backbone and base weights (`pi05_base`)
- training data (standard LIBERO demos, all four suites)
- cameras (scene + wrist)
- action space (delta EEF chunks, horizon 10, replan every 5)
- recipe and episode limits
- evaluation seed

**Training recipe.** Copied from Blackout vs Freeze, so that arm R vs N is a **direct replication** of their camera-fault contrast:

- 10k updates, batch 32, full fine-tune from `pi05_base`, bf16
- LR: linear warmup over **10k** updates to 5e-5, so evaluation falls at the end of warmup (Blackout vs Freeze Table 3). The 1k warmup in their appendix is the real-robot recipe.
- AdamW (β = 0.9, 0.95), weight decay 1e-10, gradient clip 1.0
- EMA 0.999; evaluate the EMA weights at 10k
- trained on all four LIBERO suites (1,693 episodes)
- no blackout augmentation

This is cheaper than openpi's own `pi05_libero` recipe (batch 256, 30k steps, cosine), which reaches 96.9% in-distribution against Blackout vs Freeze's 93–94%. Matching Blackout vs Freeze matters more, because their contrast is the replication anchor.

**Seeds.** Five seeds per arm: 41–45, the same in every arm. Blackout vs Freeze trained **one seed per condition**, so their +19.8 pp has no between-run uncertainty.

### 2.4 How openpi turns state on and off (checked in openpi @ `215abfb`, 2026-08-24)

- **π0.5 has exactly one state pathway: text tokens in the prompt.** The continuous `state_proj` token exists only for π0 (`models/pi0.py`, `if not self.pi05`). For π0.5, state enters only through `TokenizePrompt`, and only when `discrete_state_input=True` (`transforms.py`, `models/tokenizer.py`).
- **The stock `pi05_libero` config sets `discrete_state_input=False`.** So openpi's default LIBERO fine-tune, and the released `pi05_libero` checkpoint, use **no proprioception**. This agrees with Blackout vs Freeze, who enabled state by flipping that flag. The flag defaults to `True` for π0.5 when unset. openpi issue #687 asks why LIBERO turns it off and has had no maintainer answer since September 2025.
- **The flag is read by the data-transform factory, not by the model.** A checkpoint trained with one value and served with the other loads and runs without error, but receives a different prompt. Each arm's training and serving configs must therefore set the flag identically. Log the exact prompt string for the first observation of every evaluation.
- **Out-of-range state is not clipped.** Quantile normalization maps the 1st–99th percentiles to [−1, 1] and does not clamp. `np.digitize` then sends values above the range to bin 255, and values below it to the string `-1`, which is outside the 0–255 bin range.

  Under K-robot, arm R's prompt will therefore contain saturated or `-1` state tokens. This is the concrete mechanism by which raw state can carry the shift into the model. Check 4 (§6) logs the fraction of such tokens per arm.

## 3. Pre-registered predictions

Each entry is the predicted sign of *success(arm) − success(N)*, in percentage points. A wrist camera is present in every arm.

| Contrast | H1 any shift hurts | H6 shift type | H7 shift × encoding | T (trade-off) |
|---|---|---|---|---|
| **RK**: raw vs none, configuration shift | hurt | hurt | hurt | — |
| **CK**: corrected vs none, configuration shift | hurt | **hurt** | **help** | — |
| **RF**: raw vs none, scene blackout | hurt | help | help | — |
| **CF**: corrected vs none, scene blackout | hurt | help | help | CF < RF |
| RID / CID: in-distribution | not hurt | not hurt | not hurt | — |

Reading the table:

- **CK separates H6 from H7.**
- **RF and CF test both against H1.** They also replicate the camera-fault result, which currently rests on one paper.
- **CF − RF measures T:** does correcting the frame cost the camera-fault benefit?
- **RK is a manipulation check.** If raw state does not hurt under K, the shift is too weak to test anything (§6).

## 4. Analysis (fixed before running)

**Unit.** The trained policy (arm × seed). With 400 paired instances per condition, binomial error is about 2 pp. Between-seed variation dominates (§5).

**Estimates.** For each primary contrast, the difference in mean success between the arm and N, averaged over seeds. K pools K-robot and K-layout with equal weight per sub-condition. Sub-conditions and magnitudes are reported separately as secondary results (dose–response).

**Confidence intervals.** 95% two-level cluster bootstrap: resample seeds within each arm, then tasks within each suite, 10,000 replicates. Exact McNemar per seed on the paired episodes is secondary.

**Classification.** Same thresholds as the evidence map:

| Class | Rule |
|---|---|
| **help** | point estimate > +3 pp and the CI excludes 0 |
| **hurt** | point estimate < −3 pp and the CI excludes 0 |
| **null** | otherwise |

Holm correction across the four primary contrasts.

**T.** CF − RF with its bootstrap CI. A loss of more than 5 pp, with the CI excluding 0, counts as a trade-off.

**Decision table.** `pilot/scripts/preregistered_outcomes.py` replays every help/null/hurt combination of the four contrasts through the evidence map's planner likelihood, starting from the current posterior. The most likely patterns:

| RK | CK | RF | CF | P(pattern), current posterior | Planner winner | p(H6) | p(H7) |
|---|---|---|---|---|---|---|---|
| hurt | help | help | help | 0.23 | H7 | 0.06 | **0.94** |
| hurt | hurt | help | help | 0.12 | H6 | **0.71** | 0.29 |
| hurt | null | help | help | 0.05 | H7 (weak) | 0.29 | 0.71 |
| hurt | help | help | hurt | 0.04 | H7 + strong T | 0.06 | 0.93 |

The pilot's expected information gain is **0.43 bits**, against a prior entropy of 1.11.

Two pre-committed readings:

- **CK = null** means "corrected state removes the harm but adds nothing". It is reported as weak support for H7, not as a confirmation.
- **RK not hurt**, or **RF not help**, invalidates the corresponding half of the test. Those cells are not interpreted (§6).

The planner's likelihood (0.75 on the predicted class) is coarse. The pilot's conclusions come from the CIs above; the planner table only keeps the result comparable with Paper 4.

## 5. Power and seed count

`pilot/scripts/power_analysis.py` (Monte Carlo, conservative because it ignores pairing) gives the power to detect a difference between two arms, as *α = .05 / Holm α = .0125*:

| Seed SD | Seeds per arm | Δ = 10 pp | Δ = 15 pp | Δ = 20 pp |
|---|---|---|---|---|
| 2 pp | 3 | 0.82 / 0.47 | 0.99 / 0.83 | 1.00 / 0.98 |
| 4 pp | 3 | 0.51 / 0.21 | 0.83 / 0.49 | 0.97 / 0.75 |
| 4 pp | **4** | 0.70 / 0.41 | **0.96 / 0.79** | 1.00 / 0.97 |
| 6 pp | 4 | 0.45 / 0.20 | 0.77 / 0.48 | 0.95 / 0.76 |
| 6 pp | 5 | 0.57 / 0.30 | 0.90 / 0.67 | 0.99 / 0.92 |

Going from 400 to 1,200 episodes per condition barely changes power. **Seeds are what to buy.**

**Expected effect sizes.** The map's effects under shift are large:

- Raw state under configuration shift: median −49 pp, range −15 to −98, across 16 settings.
- Raw state under scene blackout: +17 to +20 pp.
- Corrected state under shift: +5 to +26 pp.

**Seed spread is probably larger under faults than in distribution.** The only published two-seed pairs (Blackout vs Freeze Table 14, π0.5 with blackout-augmentation recipes) differ between runs by:

| Condition | Run-to-run spread |
|---|---|
| clean | 1.4 pp |
| scene blackout | 3.4 pp |
| scene frozen | 8.7 pp |
| wrist blackout | 11.3 pp |

So the plan assumes a seed SD of about 5–6 pp under camera faults and uses **5 seeds per arm**. At 6 pp, that gives power 0.90 / 0.67 for a 15 pp effect, and 0.99 / 0.92 for 20 pp (α = .05 / Holm).

Stage 0 measures the seed SD:
- **≤ 4 pp:** drop to 4 seeds.
- **> 6 pp:** raise to 6–7 seeds, or accept that only effects of 20 pp or more are detectable.

## 6. Validity checks (run first; a failure stops interpretation)

1. **Training converged.** Every policy reaches ≥ 85% on ID. Arms are within 5 pp of each other on ID; a larger gap is reported but does not block.
2. **Camera-fault replication anchor.** R − N under F-scene, on the Spatial / Object / Goal subset, is ≥ +10 pp. Blackout vs Freeze reports +19.8.

   If this fails, the pipeline differs from theirs. One possibility is how the no-state arm omits state: Blackout vs Freeze notes that the released π0.5 LIBERO checkpoint uses no state at all. Debug before going on.
3. **Configuration shift bites.** R − N under K is < −3 pp at one or more magnitudes. If raw state does not hurt here, CK cannot discriminate. Stop and raise the magnitude, or add the height shift from 2509.18644.
4. **C is in distribution where intended.** Under K-robot, ≥ 95% of arm C's state vectors fall inside the 1st–99th percentile of training. Under R, the same statistic is expected to fail. Also log the share of state tokens that are saturated (255) or out of range (`-1`) in each arm (§2.4). This confirms the encoding does what §2.1 claims.
5. **No joint floor or ceiling.** Under each shifted condition, N and R are not *both* below 10% or *both* above 90%. N alone is expected near 9% under scene blackout (Blackout vs Freeze), so it is not tested alone. If a K condition fails, move to the next magnitude level, which is pre-listed in the config.

## 7. Budget (estimates; replace with Stage 0 measurements)

| Item | Count | Estimate |
|---|---|---|
| Fine-tunes | 3 arms × 5 seeds = 15, at 10k updates and batch 32 each | ~6–10 GPU-h each on one 80 GB GPU (openpi needs > 70 GB for a full fine-tune); ~90–150 GPU-h |
| Episodes per policy | 400 instances × (ID + 3 robot magnitudes + 2 layout levels + F-scene + F-wrist) = 3,200 | — |
| Total episodes | 15 × 3,200 = 48,000 | ~10–20 s each with batched environments; ~135–265 GPU-h |
| LP-official secondary | 1 seed per arm | ~15 GPU-h |
| **Total** | | **~250–430 GPU-h** |

## 8. Staging

| Stage | Time | Work | Exit criterion |
|---|---|---|---|
| **0: pipeline and anchor** (scripts: `pilot/stage0/`) | ~1 week | Arms N and R, seeds 41 and 42. ID + F-scene + K-robot at magnitude 0.2. Also serve the released `pi05_libero` checkpoint as a reference for arm N (§2.4), and run the `pi05_base` zero-shot exposure probe (§9). | Checks 1–3 pass. Seed SD measured; seed count fixed. |
| **1: the pilot** | ~2 weeks | All 3 arms × 5 seeds (4–7 after Stage 0) × all conditions. | Four contrasts classified. Decision table read off. Rows appended to `evidence_map/data/results.jsonl`, planner re-run. |
| **2: conditional extensions** | ~2–3 weeks | **D** (state-token dropout 0.3, regularized) if CK = help, to test whether regularization trades away RF. **A second backbone**, Diffusion Policy from scratch (not π-family), on the same grid. **A non-LIBERO replication** (RoboTwin 2.0 or 2–3 real tasks with scene-camera blackout), as §5.6 of the plan requires. | Required before any claim beyond "π0.5 on LIBERO". |

## 9. Known confounds and how they are handled

- **The π0.5 base was pretrained with state (confirmed).** The π0.5 paper (2504.16054, §IV-A) says robot proprioceptive state is discretized and fed as text tokens. Arm N therefore fine-tunes weights that saw a `State:` string in every robot example, and then never sees one.

  Blackout vs Freeze and openpi's own LIBERO recipe do the same, which keeps the anchor comparable. Two consequences:
  - Arm N is not a "state-naive" model.
  - Arms R and C differ from N partly in matching the pretraining prompt format.

  The Diffusion Policy from-scratch arm in Stage 2 removes this confound.
- **LIBERO exposure in pretraining: no evidence of it, but not officially ruled out.**
  - **Paper.** The π0.5 paper lists the pre-training mixture as:
    - mobile-manipulator home data (MM)
    - multi-environment static-arm data (ME)
    - lab cross-embodiment data including "an extended version" of Open X-Embodiment (CE)
    - high-level subtask labels (HL)
    - web data (WD)

    No simulation or LIBERO data is named. LIBERO is not part of the standard OXE release, but the "extended" OXE mix is not itemized.
  - **openpi.** It gives only "10k+ hours of robot data" for the base checkpoints. The `pi05_base` asset folder holds normalization stats for `arx`, `arx_mobile`, `droid`, `fibocom_mobile`, `franka`, `trossen`, `trossen_mobile`, `ur5e` and `ur5e_dual`, but not LIBERO. `pi05_libero` ships its own stats.
  - **Community.** One converted-checkpoint card says the base "was never fine-tuned on LIBERO". That is a third-party claim.
  - **Caveat.** openpi's `pi05_base` is a released checkpoint, not necessarily the exact model in the paper.
  - **What to write.** Treat the setting as "no known LIBERO pretraining". Say so explicitly, and add a Stage 0 probe: serve `pi05_base` zero-shot on LIBERO with the `pi05_libero` norm stats for 50 episodes per suite. Clearly above-zero success would point to exposure; near-zero is consistent with none, though weak evidence. Either way, it affects all arms equally, because they share base weights.
- **Robot-initial-state shift is also a visual shift.** The arm looks different in the image. This is inherent to a real configuration shift and is the same for every arm. It is why K pools robot and layout, and why the sub-conditions are reported separately.
- **Relative state can leak episode progress.** It can act as a timer, the copycat route. The F-wrist after-closure condition and per-step action logs allow a check.
- **Episode-start faults leave the state almost constant.** Early in an episode the state carries little information, so F-scene mainly tests whether state helps *localize the arm*. F-wrist after closure tests mid-task use.

## 10. Outputs

- **`pilot/results/episodes.parquet`:** one row per episode, with arm, seed, instance, condition, magnitude, success, steps, fault onset and the physical indicators logged by the Blackout vs Freeze harness, if adopted.
- **`pilot/results/contrasts.json`:** the four primary contrasts and T, with CIs and classes.
- **New rows in `evidence_map/data/results.jsonl`:**
  - comparison `presence`, shift `robot_init` / `spatial` / `visual_fault`
  - state frame `absolute` / `relative`
  - provenance `pi0.5` + `LIBERO`

  Then re-run `evidence_map/scripts/run_planner.py`.
