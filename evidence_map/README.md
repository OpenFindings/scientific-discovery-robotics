# Evidence map: does proprioceptive state help or hurt robot manipulation policies?

*Step 5.1 of `research/area_selection_from_paper3_paper4.md`. Built 2026-10-15; updated the same day after screening all flagged papers.*

## Bottom line

*Updated 2026-10-15 after all 89 citation-flagged papers were screened (§1).*

**No single variable explains the contradiction, even after the update.** The pre-set rule (§3) still fails: the best variable, evaluation shift, now reaches 0.78 paper-weighted accuracy against a 0.90 bar. Per the plan, **we still do not switch to the second-choice area.** But the extra reading changed the picture in three ways:

1. **"Shift" was the wrong unit. The *type* of shift sets the sign.**
   - **Configuration shift** (object, table or base position, embodiment, or the robot's starting joints outside the demonstrated range). Raw state (absolute frame, naive fusion) hurt in **16 of 16 settings across 6 papers**, now including D3P (robot initial state, −22.5 pp) and Mirage (unaligned state on a new arm, −40 to −45 pp).
   - **Camera faults** (blackout, frozen frame, robot masked in the image, with the robot itself in distribution). Raw state *helped* in **11 of 14 settings**, by up to +69 pp, with no losses (Blackout vs Freeze, π0.5 and GR00T).

   So "state hurts out of distribution" is half true. It hurts when the robot's own configuration is off-distribution, and it rescues the policy when the cameras fail.
2. **The harm under configuration shift is not fixed.** It reverses when state is re-expressed in a frame consistent across settings (relative, image-grounded, or Mirage's cross-embodiment alignment) or when fusion is regularized: 8 help, 3 hurt, 1 null. But only **three papers** provide these cells, and two of them have a single setting each.
3. **The residual contradiction is in-distribution and small.** With raw state and in-distribution tests there are 15 help, 14 hurt and 31 null settings from 14 papers. Most effects there are within ±3 pp. This residual is the only thing still failing the rule.

**Planner (§4).** Two hypotheses were added after reading the new papers:
- **H6:** the type of shift sets the sign.
- **H7:** shift type × state encoding sets the sign.

H7 now leads (posterior 0.67, and 0.55–0.68 with any one paper left out), ahead of H6 (0.28). The original single-factor H1, which says any shift hurts, drops from 0.97 to 0.05. The most informative experiments are all **configuration-shift cells with corrected frames or regularized fusion**. Those cells hold 0–1 papers each, and they are exactly where H6 and H7 disagree.

The open, resolvable question is now:

> *Under configuration shift, does relative / grounded / regularized state remove the harm of proprioception **while keeping its benefit under camera faults**?*

No paper tests both halves on one policy.

## 1. Retrieval: citation graph, not keywords

`scripts/harvest_citation_graph.py` collects both directions of the citation graph from Semantic Scholar:

- **Seeds:** papers that cite each seed, and papers each seed cites. The seeds are the three conflicting papers (2509.18644, 2608.03052, 2609.02546) and the two papers that introduced the shortcut hypothesis: de Haan 2019 (1905.11979) and Wen 2020 (2010.14876).
- **Second hop:** the same collection for five first-hop papers that themselves run state ablations (2602.12032, 2605.13067, 2607.07101, 2601.16667, 2506.13679).

| Stage | Count |
|---|---|
| Unique papers in the citation neighbourhood | 912 |
| Flagged by a loose title/abstract screen (`scripts/screen_pool.py`, high recall) | 82 |
| Screened (all flagged papers, plus citation-retrieved classics such as Octo, OpenVLA-OFT, robomimic, UMI) | 89 |
| Contributing numeric with-vs-without-state results | **14 papers, 123 settings** (was 9 papers, 75 settings) |
| Contributing related contrasts (state terms, frame, fusion) or qualitative results | 7 papers with rows; 16 more recorded as context in `data/screening.jsonl` |
| Excluded after reading (driving, locomotion, Atari, datasets, no state arm) | 49 |
| Duplicates (title variants of included papers) | 2 |
| Unavailable (no full text reachable) | 3 |

**How the 69 previously unread papers were screened.**
- **Off-topic at abstract level (22): excluded.** These were driving, locomotion, Atari/gaze, RL causal-confusion theory, datasets, pose estimation and VLM captioning.
- **Remaining 47:** 2 were duplicates of included papers. Full text was sought for the other 45 (alphaXiv, with web search as fallback); 3 could not be retrieved. Reasons for each paper are in `data/screening.jsonl`.

**Five new papers add with-vs-without-state rows:**
- **2609.39145 Blackout vs Freeze:** π0.5 and GR00T, camera faults.
- **2511.00555 D3P:** Diffusion Policy, robot initial-state shift.
- **2402.19249 Mirage:** BC-RNN and Diffusion Policy, embodiment shift, aligned vs unaligned state.
- **2512.20166 LoLA:** in-distribution LIBERO.
- **2508.08113 AimBot:** π0 with zeroed state, in-distribution LIBERO.

**Two papers add fusion rows:**
- **2610.07015 LEAP:** keeping state out of an auxiliary decoder adds +12 to +18 pp.
- **2610.14854 DEFT-VLA:** state-token masking adds +55 pp on a real contact task.

**Unavailable:**
- "Restoring the Right Stream" (OOD robustness on LIBERO-Plus).
- ARViP (IEEE TII 2026).
- A 2022 vision-proprioception pushing RL paper.

These need manual retrieval.

## 2. Recorded settings

`data/results.jsonl` has one row per reported comparison, with the table or figure it came from. `effect_pp` is success with state minus success without state, so **negative means state hurt**. Each row records:

- shift (in-distribution, spatial, embodiment, perturbation, robot initial state, or camera fault)
- state frame (absolute, relative, image-grounded, or cross-embodiment aligned)
- whether velocity terms are included, and the state-history length
- fusion (naive, or regularized/grounded)
- action space
- whether a wrist camera is present
- platform (sim or real), backbone, and whether it is pretrained
- benchmark
- masking protocol

`scripts/build_records.py` regenerates the file and documents the coding.

Summary by paper, for with-vs-without-state comparisons only:

| Paper | Setting | Effect of state (pp) |
|---|---|---|
| 2509.18644 State-free policies (π0, ACT, DP; real) | in-distribution | 0, +3 |
| | height / horizontal / embodiment shift | −37 to −98 (9 settings) |
| 2608.03052 State interfaces (π0.5, RoboCasa) | in-distribution | −0.2 to +6; history ×8 gives +10.6 |
| 2602.12032 GAP (BC, Octo; sim + real) | naive, in-distribution | −10 (sim), −20 (real), −10 (Octo) |
| | naive, position shift | −15, −20 |
| | GAP-regularized | +8 (in-distribution), +11 (shift) |
| 2605.13067 Absolute vs relative (ACT, real rail) | absolute state | −15 (in-distribution), −20 (shift) |
| | episode-relative state | +30 (in-distribution), +5 (shift) |
| 2607.07101 GeoProp (DP, π0; sim + real) | naive, sim | −5.4 to +2.3 |
| | naive, real single view | +10 (DP), +1.3 (π0) |
| | grounded | +6 to +21 |
| 2601.16667 ReViP (π0) | in-distribution | +3.7 (LIBERO), +30 (real) |
| | state–vision mismatch recovery | −58 |
| 2506.23944 Adapt Your Body (BC, robomimic/MimicGen) | all state terms | −50 to +12; velocity terms drive the losses |
| | noise-regularized | −1 to +12 |
| 2409.20537 HPT (real, from scratch) | in-distribution | +16.6 |
| 2602.06575 ThinkProprio (FLOWER, CALVIN) | in-distribution | −5.8 to +0.8 |
| **2609.39145 Blackout vs Freeze (π0.5, GR00T; LIBERO, n=450)** | clean, in-distribution | −1.3, +1.6 |
| | scene/wrist camera blackout | +11 to +20 |
| | frozen frame | 0 to +5.6 |
| | robot masked in scene/wrist view | +30 to +69 (objects masked: 0) |
| **2511.00555 D3P (Diffusion Policy, RLBench)** | fixed initial joints | +3.4 |
| | initial joints ±10° | −22.5 |
| **2402.19249 Mirage (BC-RNN sim; DP real)** | in-distribution | −7 to +34 (sim), +7.5 (real) |
| | new arm, aligned state + cross-painted images | +4 to +26 |
| | new arm, aligned state, raw images | −8 to +2 |
| | new arm, unaligned state (real) | −40, −45 |
| **2512.20166 LoLA (Qwen2.5-VL VLA; LIBERO)** | in-distribution | +2.5 to +10.5 |
| **2508.08113 AimBot (π0; LIBERO)** | in-distribution | +0.2 to +3.0 |

Related contrasts outside the main test:

- **robomimic:** adding velocity or joint terms costs 1–32 pp.
- **UMI:** relative inter-gripper pose adds +40 pp in bimanual tasks.
- **ZETA:** relative instead of absolute state adds +11 to +13 pp under embodiment shift.
- **RRDF:** delayed fusion adds +18 pp in sim and +52 pp under real position shift.
- **Octo:** "generally worse" with state, no numbers.
- **LEAP:** feeding state to an auxiliary geometry decoder costs 12–18 pp (a "proprioceptive shortcut").
- **DEFT-VLA:** a state-token masking curriculum lifts a real contact task from 15% to 70%.
- **Context only (no rows):**
  - A planner-goal "displacement-integrator shortcut" collapses π0.5 to 15% (2609.30833).
  - GAM finds history H=1 more robust than H=2/4 on LIBERO-Plus.
  - TwinBrainVLA turns state off in all sim benchmarks.

## 3. Single-variable test (`scripts/single_variable_test.py`)

**Rule, fixed before running.** The test uses with-vs-without-state records. Each record is classed as hurt (< −3 pp), null, or help (> +3 pp), and each paper's records carry a combined weight of 1. Each level of a variable predicts the weighted-majority class within that level. A variable *explains* the contradiction if both conditions hold:

1. paper-weighted accuracy on non-null records is at least 0.90;
2. no level has counterexamples from two or more papers.

| Variable | Accuracy (was) | Explains? |
|---|---|---|
| Evaluation shift (now 6 levels) | 0.781 (0.733) | no |
| Shift: any vs none | 0.713 | no |
| Action space | 0.681 (0.658) | no |
| State history > 1 | 0.639 | no |
| State frame / fusion / velocity / pretrained / platform / wrist camera | 0.608 | no |
| Best pair: shift × state frame | 0.823 (0.770) | no |
| *Post hoc:* shift group (in-distribution / configuration / camera fault) × frame corrected × fusion | 0.837 | no |

**Why it still fails.** Every *shifted* level is now clean or nearly clean:
- **configuration shift, raw state:** hurt 16/16
- **configuration shift, corrected or regularized state:** help 8/12; the 3 hurts are Mirage rows where the images are also broken
- **camera fault:** help 11/14, no hurt

The failing level is **in-distribution with raw state**: 15 help / 14 hurt / 31 null. Its counterexamples come from five papers. The test cannot pass while those cells are pooled. No recorded variable splits them:
- velocity terms: 1 paper
- history > 1: 2 papers, mixed
- wrist camera: counterexamples on both sides

The two post-hoc variables are reported separately in `results/single_variable_test.json` (`posthoc`), because they were defined after seeing the new data.

## 4. Loaded into the Paper 4 planner (`scripts/run_planner.py`)

**Question:** why do reported effects of state conflict in sign? Seven exclusive hypotheses, with uniform priors:

| Hypothesis | Claim | Cells correct |
|---|---|---|
| H1 evaluation shift | Any shift: state hurts. In distribution: neutral or helpful. | 31/43 |
| H2 frame | Absolute frames cause the harm; corrected (relative, grounded or aligned) state helps. | 16/43 |
| H3 fusion | Naive fusion lets state dominate; regularized fusion makes state help. | 17/43 |
| H4 observability | State helps when cameras miss task information: no wrist view, or a camera fault. | 24/43 |
| H5 heterogeneous | Effects are task-specific noise. | 12/43 |
| *H6 shift type (post hoc)* | Configuration shift: hurt. Camera fault: help. In distribution: not hurt. | 32/43 |
| *H7 shift × encoding (post hoc)* | Under configuration shift, raw state hurts and corrected or regularized state helps. Camera fault: help. In distribution: not hurt. | 34/43 |

**Graph.** There are 43 claim nodes, one per (paper × shift group × fusion × frame × outcome) cell. Each claim derives from its paper, and each paper derives from provenance roots: its backbone family and its benchmark or setup. A cell is linked to hypotheses only when they disagree about it.

**Results.**

- **The stock gap analysis still flags no gaps.** The thresholds count edge volume and provenance. They cannot see that the cells separating H6 from H7 are almost empty. This is the same limitation as before and is worth feeding back into Paper 4.
- **Bayesian replay of all cells** gives these posteriors:

  | Hypothesis | Posterior |
  |---|---|
  | H7 | **0.674** |
  | H6 | 0.279 |
  | H1 | 0.046 |
  | All others | < 0.001 |

  With any one paper left out, H7 still wins (0.55–0.68). The paper whose removal moves it most is Blackout vs Freeze: without it, H1 recovers to 0.23. The *camera-fault half* of the story therefore rests on a single paper. H7 was formulated after seeing the data, so its lead is a hypothesis to test, not a finding.
- **Cell-level preposterior ranking** (Paper 4's expected-information-gain term):

| Expected information gain (bits) | Design cell | Existing papers |
|---|---|---|
| 0.318 | config shift / corrected frame / regularized / wrist or no wrist | none |
| 0.318 | config shift / absolute / regularized / wrist | none |
| 0.318 | config shift / corrected / naive / wrist | 2605.13067 (1 setting) |
| 0.318 | config shift / corrected / naive / no wrist | 2402.19249 (Mirage) |
| 0.318 | config shift / absolute / regularized / no wrist | 2602.12032 (1 sim setting) |
| 0.070 | camera fault / any encoding | 2609.39145 only, naive absolute only |

Outputs are in `results/single_variable_test.json`, `results/planner_output.json` and `results/planner_graph.json`.

## 5. What this means for the plan

- **Keep area A; do not switch.** The kill check still did not fire, and the question is now sharper and still open.
- **Redesign the pilot (step 5.3) as a 3 × 2 design on one backbone with fixed data and cameras.**
  - **State encoding (3 levels):** none; raw absolute with naive fusion; corrected or regularized (relative frame plus GAP/NADA-style regularization, or image-grounded).
  - **Test condition (2 levels):** a configuration shift (object position plus robot initial joints) and a camera fault (scene blackout or robot occlusion).
  - **In-distribution:** a baseline only.
  - **What it decides:** this is the cheapest design that separates H6 from H7. It also checks the untested trade-off: does correcting the encoding **keep** state's camera-fault benefit, or does making the policy rely less on state throw that benefit away?
  - **Suggested setup:** LIBERO-Plus already provides both "robot initial state" and "camera" perturbations, and D3P and Blackout vs Freeze each used one of them.
- **Do not spend effort re-showing that raw state hurts under configuration shift.** That is now 6 papers and 16/16 settings.
- **The in-distribution residual is a separate, low-value question.** Effects there are mostly within ±3 pp, and no recorded variable splits them. Treat "state is roughly neutral in distribution" as the working assumption.
- **Provenance is still concentrated.**
  - The camera-fault evidence is one paper on LIBERO.
  - The corrected-frame-under-shift evidence is three papers.
  - A real-robot camera-fault result, or any second camera-fault paper, is the cheapest way to strengthen the map.

## 6. Simulation analogue (not literature evidence; kept out of `results.jsonl`)

`research/sim_findings_cpu_libero.md` ran the pilot's shifts in real LIBERO with small blind and oracle-perception policies. The ego-centric oracle (perfect wrist-camera-like object offsets) shows two things:
- **Robot-initial-state shift:** state hurts (−20 pp), and the episode-relative frame does not fix it (−22 pp).
- **6 cm object-layout shift:** state helps (+11 / +18 pp).

That splits "configuration shift" into two sub-types with opposite expected signs. Published harm under object-position shifts would then reflect imperfect visual grounding (vision suppression) rather than state per se. A blind MLP with absolute state solves 71.5% of in-distribution LIBERO, which supports treating in-distribution results as uninformative.

## Caveats

- The settings were coded by one reader (Claude), from paper tables. No second coder checked them.
- Effects are in percentage points across very different tasks; pooling only works at the sign level.
- ReViP's masking protocol is unclear, and 2605.13067 changes the action space together with the state encoding.
- The hypothesis predictions in `run_planner.py` are coarse (one outcome class per cell). The likelihoods (0.75 on match) are a modelling choice, not fitted.
- Three flagged papers could not be retrieved (§1). The 22 papers excluded on abstract alone were clearly off-topic, but no one read them in full.
- H6 and H7 were added after the new papers were read. Their posteriors show which explanation fits best; they are not confirmatory evidence.
- Mirage's simulation rows treat robosuite's built-in cross-arm EEF frame as "aligned". Its real-robot rows report alignment explicitly.
- In this update, 28 of the full-text reads were done by parallel reading agents and the rest by Claude directly; Claude coded every row. No independent human second coder has checked them yet.
