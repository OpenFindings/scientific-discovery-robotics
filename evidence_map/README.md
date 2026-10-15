# Evidence map: does proprioceptive state help or hurt robot manipulation policies?

*Step 5.1 of `research/area_selection_from_paper3_paper4.md`. Built 2026-10-15.*

## Bottom line

**No single variable explains the contradiction.** The test (§3) was fixed before it was run, and no variable meets its bar. Per the plan, **we do not switch to the second-choice area.** The map does change the question, though:

- **Shifted evaluation (object, table or robot-base position, embodiment, or an injected state–vision mismatch).** Naive, absolute-frame state hurt in **13 of 13** settings across 4 papers. This is the source of the widely cited "0 → 85%" result. The one shifted exception (+5 pp) used episode-relative state.
- **In-distribution evaluation.** Results are split: 11 help, 14 hurt, 20 null, across 9 papers. The evaluation-shift variable cannot explain this split. No other recorded variable does either: the best single variable reaches 0.73 paper-weighted accuracy against a 0.90 bar.
- **Paper 4 planner, once the evidence is loaded.** It strongly prefers the evaluation-shift hypothesis: posterior 0.97, and 0.88–0.97 when any one paper is left out. Its most informative next experiments are all in cells with **zero existing papers**. Those cells are with-vs-without-state comparisons **under spatial or embodiment shift, using relative, grounded or regularized state.**

So about half of the apparent contradiction comes from comparing in-distribution results with shifted ones. The open, resolvable question is narrower than the one in the plan:

> *Does the out-of-distribution harm from proprioceptive state disappear when state is relative, image-grounded, or regularized?*

No paper has tested this directly. Relative or regularized state under shift has been tried in only two papers, and only against a narrow set of baselines.

## 1. Retrieval: citation graph, not keywords

`scripts/harvest_citation_graph.py` collects both directions of the citation graph from Semantic Scholar:

- **Seeds:** papers that cite each seed, and papers each seed cites. The seeds are the three conflicting papers (2509.18644, 2608.03052, 2609.02546) and the two papers that introduced the shortcut hypothesis: de Haan 2019 (1905.11979) and Wen 2020 (2010.14876).
- **Second hop:** the same collection for five first-hop papers that themselves run state ablations (2602.12032, 2605.13067, 2607.07101, 2601.16667, 2506.13679).

| Stage | Count |
|---|---|
| Unique papers in the citation neighbourhood | 912 |
| Flagged by a loose title/abstract screen (`scripts/screen_pool.py`, high recall) | 82 |
| Read in full (flagged papers plus citation-retrieved classics such as Octo, OpenVLA-OFT, robomimic, UMI) | 20 |
| Contributing numeric with-vs-without-state results | 9 papers, 75 settings |
| Contributing related contrasts (state terms, state frame, fusion) or qualitative results | 5 papers |
| Excluded after reading (reasons in `data/screening.jsonl`) | 6 |

The search did not reach "every result". 69 flagged papers are still unread. Most are driving or causal-confusion offshoots. The most likely relevant ones form the next reading queue:

- **State history and memory:** 2505.09561, 2510.00695, 2510.00726, 2606.16447, 2609.27247
- **Shortcut mitigation:** 2207.09705, 2610.04616, 2609.12641
- **Other sensors and embodiments:** 2501.04693, 2604.07993

## 2. Recorded settings

`data/results.jsonl` has one row per reported comparison, with the table or figure it came from. `effect_pp` is success with state minus success without state, so **negative means state hurt**. Each row records:

- shift (in-distribution, spatial, embodiment, or perturbation)
- state frame (absolute, relative, or image-grounded)
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

Related contrasts outside the main test:

- **robomimic:** adding velocity or joint terms costs 1–32 pp.
- **UMI:** relative inter-gripper pose adds +40 pp in bimanual tasks.
- **ZETA:** relative instead of absolute state adds +11 to +13 pp under embodiment shift.
- **RRDF:** delayed fusion adds +18 pp in sim and +52 pp under real position shift.
- **Octo:** "generally worse" with state, no numbers.

## 3. Single-variable test (`scripts/single_variable_test.py`)

**Rule, fixed before running.** The test uses with-vs-without-state records. Each record is classed as hurt (< −3 pp), null, or help (> +3 pp), and each paper's records carry a combined weight of 1. Each level of a variable predicts the weighted-majority class within that level. A variable *explains* the contradiction if both conditions hold:

1. paper-weighted accuracy on non-null records is at least 0.90;
2. no level has counterexamples from two or more papers.

| Variable | Accuracy | Explains? |
|---|---|---|
| Evaluation shift | 0.733 | no |
| Action space | 0.658 | no |
| Wrist camera | 0.637 | no |
| Fusion | 0.589 | no |
| State history > 1 | 0.587 | no |
| State frame | 0.584 | no |
| Velocity terms / pretrained / platform | 0.542 | no |
| Best pair: shift × state frame | 0.770 | no |

**Why evaluation shift fails.** Its "shifted → hurt" level is nearly clean. With naive fusion it holds in 13/14 settings, and the one exception is relative state. The in-distribution level is the problem: it has hurt counterexamples from five papers (2506.23944, 2602.12032, 2605.13067, 2607.07101, 2602.06575). Within in-distribution, naive-fusion results, no variable passes either. The leads are velocity terms (1 help / 6 hurt / 2 null) and state history in concatenation (1 / 3 / 0), but both are thin.

## 4. Loaded into the Paper 4 planner (`scripts/run_planner.py`)

**Question:** why do reported effects of state conflict in sign? Five exclusive hypotheses, with uniform priors:

| Hypothesis | Claim |
|---|---|
| H1 evaluation shift | Evaluation shift sets the sign. |
| H2 frame | Absolute frames and velocity terms cause the harm; relative or grounded state helps. |
| H3 fusion | Naive fusion lets state dominate training; regularized fusion makes state help. |
| H4 observability | State helps when cameras miss task information. |
| H5 heterogeneous | Effects are task-specific noise. |

**Graph.** There are 27 claim nodes, one per (paper × evaluation × fusion × frame × outcome) cell. Each claim derives from its paper, and each paper derives from provenance roots: its backbone family and its benchmark or setup. Each hypothesis predicts an outcome class per cell. A cell supports the hypotheses whose prediction it matches and contradicts the others. A cell is linked only when the hypotheses disagree about it; if every hypothesis predicts the same outcome, the observation discriminates nothing.

**Results.**

- **The stock gap analysis flags no gaps.** Every hypothesis clears its thresholds: at least two edges, enough support confidence, and more than one provenance root. The thresholds count volume. They cannot see that the *discriminating* cells are empty. This is a concrete limitation to feed back into Paper 4.
- **Bayesian replay of all cells** through the planner's `BayesianBeliefUpdater` gives a posterior of H1 **0.972**, H4 0.016, H3 0.011, H2 0.002, H5 0.000. H1 matches 20 of 27 cells. With any one paper left out, H1 still wins every time (0.88–0.97). The pooled posterior is overconfident because cells from one paper are not independent; the ranking is the robust part.
- **Cell-level preposterior ranking.** This applies Paper 4's expected-information-gain term to design cells instead of to the generic action catalogue. The top cells are:

| Expected information gain (bits) | Design cell | Existing papers |
|---|---|---|
| 0.044 | shifted / relative / regularized / no wrist | none |
| 0.044 | shifted / grounded / regularized / no wrist | none |
| 0.041 | shifted / absolute / regularized / no wrist | 2602.12032 (sim only) |
| 0.028 | shifted / relative or grounded / naive / no wrist | none |
| 0.027 | shifted / relative or grounded / regularized / wrist | none |

Outputs are in `results/single_variable_test.json`, `results/planner_output.json` and `results/planner_graph.json`.

## 5. What this means for the plan

- **Keep area A; do not switch to the world-model evaluator topic.** The single-variable kill check did not fire.
- **Reframe the question.** Out-of-distribution harm from naive absolute state is already well established (4 papers, 13/13 settings). Testing it again adds little. The decisive, untested comparison is with versus without **relative, grounded or regularized** state, under spatial and embodiment shift, with the backbone, data and cameras fixed. That is the core of the pilot in step 5.3. Its factorial design should be cut down to these cells.
- **The in-distribution residual is a separate, smaller question.** Velocity terms, concatenated state history, and low demonstration counts are the leads. Adapt Your Body already supports velocity terms on robomimic/MimicGen. A cheap re-analysis can check this on released checkpoints.
- **Watch for provenance concentration.** Six of the nine papers use π0/π0.5, Octo, or robomimic/MimicGen-family setups. A real-robot result on a non-π backbone is worth more than another LIBERO number.

## Caveats

- The settings were coded by one reader (Claude), from paper tables. No second coder checked them.
- Effects are in percentage points across very different tasks; pooling only works at the sign level.
- ReViP's masking protocol is unclear, and 2605.13067 changes the action space together with the state encoding.
- The hypothesis predictions in `run_planner.py` are coarse (one outcome class per cell). The likelihoods (0.75 on match) are a modelling choice, not fitted.
- 69 citation-flagged papers are still unread (§1).
