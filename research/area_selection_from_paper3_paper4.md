# Choosing a robotics / physical-AI research area using Paper 3 and Paper 4

*Date: 2026-10-15. Literature snapshot: arXiv through mid-October 2026.*

## 0. Summary

**Top choice.** The proprioceptive-state and action-frame question in
generalist robot policies (VLAs), joined to the cross-embodiment transfer
debate. The core question:

> *When a VLA fails to generalize spatially or across embodiments, is the
> cause (a) proprioceptive state used as a shortcut, (b) an
> embodiment-specific state/action frame, (c) incomplete visual observation,
> or (d) an evaluation-protocol artifact (target embodiment already seen in
> pretraining)?*

It is the only candidate that passes every screen derived from Papers 3 and 4:

- **Explicit contradiction.** Papers report opposite signs for the same
  intervention.
- **Quantified.** Effects are given in percentage points.
- **Recurrent.** The question has come back from 2019 to 2026.
- **Broad cluster.** At least four independent papers are involved.
- **Competing hypotheses.** They can be listed explicitly.
- **Cheap resolution.** The resolving action is a low-cost discriminating
  re-analysis on released checkpoints, the action type Paper 4 found to
  resolve questions historically.

**Second choice.** Calibration of world models used as robot policy
evaluators. The topic gets heavy attention and has quantified, contestable
claims. It is more crowded, and resolving it needs paired real-robot rollouts.

**Avoid as a standalone question:**

- **"Benchmarks are broken" critiques.** These are methodological challenges,
  and Paper 3 found 0 of 61 such clusters resolved. Use the audit diagnostics
  as tools instead.
- **Tactile-VLA.** Many sensors with no shared instrument (Paper 3's
  instrument-diversity penalty), little review recognition, and few direct
  contradictions.
- **Egocentric-data and RL-post-training scaling.** The questions are real,
  but large labs own them and only large compute can resolve them.

Section 5 lists the further research to do before committing.

---

## 1. Turning Paper 3 and Paper 4 into a screening rubric

Paper 3 (n=980 questions, five cutoffs) and Paper 4 (planner over 374 tension
questions) produce two different objectives. A research area should score on
both.

| # | Screen | Source evidence | Objective |
|---|---|---|---|
| S1 | **Recognized in recent reviews or surveys** | Paper 3: review overlap β=+0.34, the strongest and most stable predictor | Attention |
| S2 | **Dense direct evidence** (≥4 independent papers bear on the exact question) | Paper 3: β=+0.26; 4-paper clusters addressed at 0.54 vs 0.32 | Attention |
| S3 | **Explicit competing hypotheses** | Paper 3: β=+0.17 on attention, β=+0.56 (p=0.08) on answered-given-attention; Paper 4 makes hypotheses mandatory | Both |
| S4 | **Explicit contradiction, not methodological doubt** | Paper 3: contradiction β=+0.30; method-challenge clusters resolved **0/61** | Resolution |
| S5 | **Quantified and recurrent tension** | Paper 3: all refutations came from quantified clusters and all resolutions from recurrent ones (case-level evidence: one physical refutation event) | Resolution |
| S6 | **Low instrument / platform diversity required** | Paper 3: β=−0.21; multi-instrument questions are synthesis problems that no single group owns | Attention |
| S7 | **Tractable with existing "archival" assets** (open checkpoints, datasets, simulators) | Paper 3 Cox model; Paper 4's calibrated configuration raises tractability. The cheap discriminating re-analysis resolved the case, not the expensive new observation | Resolution |
| S8 | **Provenance-concentrated evidence** (many results trace back to one dataset, backbone or benchmark) | Paper 4: novelty = provenance independence; the gap type "no independent corroboration" gives the highest-value actions | Resolution |

Two cautions carry over from Paper 4:

- **Attention and resolvability pull in opposite directions on breadth.**
  Broad clusters attract the community, but the planner sees their evidence as
  already saturated. The best target is therefore a **broad cluster whose
  evidence is provenance-concentrated** (S2 + S8). The topic is visible, but
  its claims all rest on one data source or protocol, so one independent check
  carries a lot of information.
- **Specificity is under-measured, not bad.** Paper 4 re-adjudicated the
  high-specificity stratum and found 69–94% of questions labeled "unaddressed"
  had in fact been engaged. A narrow, object-level question is fine if the
  outcome is measured through the citation graph rather than abstract keyword
  matching.

The rubric is qualitative. It applies astronomy-fitted effects
(AUC ≈ 0.71, κ ≈ 0.2 label ceiling) to another field without refitting.
Section 5.4 proposes validating it on robotics directly.

---

## 2. Robotics-specific translation

| Paper 3 / 4 concept (astronomy) | Robotics / physical-AI analog |
|---|---|
| Instrument / facility | Robot embodiment, sensor suite, simulator, benchmark |
| Archival data | Open X-Embodiment, DROID, BridgeData V2, open π0/π0.5/OpenVLA checkpoints, LeRobot, RoboCasa, RoboTwin 2.0, WMBench |
| Discriminating re-analysis | Re-evaluating released checkpoints under controlled ablations or shifts |
| New observation (expensive) | Large new real-robot data collection or fleet runs |
| Provenance root | Shared base model (PaliGemma/π0 family), shared dataset (OXE/Bridge), shared benchmark (LIBERO/SimplerEnv) |
| Review overlap | 2026 surveys: world models (2605.00080, 2606.00113, 2601.07823), embodiment gap (2608.18433), data pyramid (2607.24744), human video (2604.27621) |
| Premise refuted | A widely adopted design default overturned, e.g. "always feed proprioception" or "LIBERO SOTA = capability" |

Provenance concentration is unusually strong in robotics right now. A large
share of 2025–26 VLA results fine-tune the same few backbones and report on
LIBERO or SimplerEnv. An audit (2606.04233) found 79 LIBERO papers in March
2026 alone. The same audit found that a 0.09B probe with no language encoder
scores at SOTA on LIBERO, and that only 19.8% of LIBERO SOTA claims are
provably significant. By Paper 4's logic, much of the field's evidence has a
single root. One independent, well-controlled check is therefore worth more
here than in a mature field.

---

## 3. Candidate areas scored against the rubric

Scores: ● strong, ◐ partial, ○ weak or absent.

| Area | S1 Recog. | S2 Dense | S3 Hyp. | S4 Contra. | S5 Quant+Recur | S6 Low-div | S7 Tractable | S8 Prov.-conc. | Read |
|---|---|---|---|---|---|---|---|---|---|
| **A. Proprioceptive state / action frame in VLAs** | ◐ | ● | ● | ● | ● | ● | ● | ● | **Best resolution profile** |
| **B. Cross-embodiment zero-shot transfer** | ● | ● | ● | ◐ | ● | ○ | ◐ | ● | **Best attention profile** |
| C. World models as policy evaluators | ● | ● | ● | ◐ | ● | ◐ | ◐ | ◐ | Strong but crowded |
| D. Benchmark validity (LIBERO/CALVIN/Simpler) | ◐ | ● | ◐ | ○ (method doubt) | ● | ● | ● | ● | Use as tool, not as the question |
| E. Egocentric human-data scaling | ● | ● | ◐ | ◐ | ◐ | ◐ | ○ (compute) | ◐ | Large labs own it |
| F. RL post-training of VLAs | ◐ | ● | ◐ | ◐ | ◐ | ● | ◐ | ◐ | Hot, still forming |
| G. Tactile / force in VLAs | ○ | ◐ | ○ | ○ | ○ | ○ (sensor zoo) | ○ | ○ | Low on both objectives |

### A. Proprioceptive state and action representation (top resolution target)

The contradiction is explicit and quantified:

- **State hurts.** In *Do You Need Proprioceptive States?* (2509.18644),
  removing state with relative-EEF actions and full wrist-camera coverage
  raises height generalization from 0% to 85% and horizontal generalization
  from 6% to 64% across π0, ACT and Diffusion Policy. State noise and diverse
  data do not fix the state-based policies.
- **State helps (end-effector frame).** ZETA (2609.02546) finds that
  EEF-Delta *state* improves strict zero-shot cross-embodiment transfer by
  about 15 pp over absolute EEF state. On real humanoid targets it goes from
  0% to 80–100% task progress. ZETA treats state-free as a degenerate EEF-Delta
  state, which reconciles part of the dispute.
- **State helps (controlled study).** *How Should VLAs Use Proprioceptive
  State?* (2608.03052) shows current state gives a small gain (+3.1 pp, the
  only CI that excludes zero). An 8-frame history routed to the action head
  adds +10.8 pp on composite tasks, and a slot-matched control confirms the
  gain. Long raw history hurts precision tasks.
- **Recurrence.** The shortcut / causal-confusion hypothesis goes back to
  De Haan et al. 2019 and Wen et al. 2020, and was re-mined in 2025 and 2026.
  That fits Paper 3's "recurrent across cutoffs" profile.
- **Provenance concentration.** The main results share the π0/π0.5 backbone
  family. Their evaluation settings differ (real height/horizontal shift,
  sim embodiment shift, RoboCasa in-distribution). The contradiction may come
  from *setting*, not *state*, and that is checkable.

Competing hypotheses, ready for a Paper 4-style hypothesis set:

1. **H-shortcut.** Absolute state lets the policy memorize trajectories.
   Removing it helps only under spatial shift.
2. **H-frame.** The harm comes from the frame, not from state. EEF-relative
   state should keep the benefit and lose the harm.
3. **H-observability.** State is redundant when cameras give full task
   observation. It helps when vision is partial. This predicts an interaction
   with camera field of view.
4. **H-protocol.** Apparent gains or losses come from in-distribution versus
   shift evaluation, or from target-embodiment exposure.

This is a narrow, object-level question. Paper 3 predicts it gets *measured*
attention more slowly. Paper 4 shows that is mostly a measurement artifact,
and it is exactly where questions resolve.

### B. Cross-embodiment transfer (top attention target, couples with A)

- Recognition is very high: a dedicated 2026 survey (2608.18433) and a stream
  of LAP, ZETA, X-VLA and latent-action papers.
- ZETA gives the key reconciling fact. Adding just 5% target-embodiment data
  to pretraining raises transfer by 13.4 pp. Its re-classification shows Octo,
  OpenVLA, π0.7 and Gemini Robotics 1.5 "zero-shot" claims are
  *pretrain-exposed*, while LAP and RDT2 are *strict*. Mixed reports of
  transfer versus negative transfer may therefore partly be a protocol
  artifact. That is a contestable, quantified claim that a meta-re-analysis
  can test.
- Risk: high platform diversity (S6). As a whole, the area is a synthesis
  problem. Join it to A to get a single-group-ownable entry point.

### C. World models as policy evaluators (second choice)

- Three or more 2026 surveys and a CVPR 2026 challenge mean the area is
  recognized and crowded.
- Quantified tensions exist. GigaWorld-1 (2607.02642) shows visual fidelity
  correlating ρ=0.78 with evaluator quality while "appearance stability"
  correlates *negatively* (ρ≈−0.44), and generated rollouts *overestimate*
  success by up to +0.2–0.3 on contact-rich subtasks.
- Competing hypotheses (visual fidelity vs. action-faithfulness vs.
  long-horizon memory) are explicit.
- Tractable via the public WMBench data (2,989 paired trajectories). A narrow
  question would fit the resolution profile: *is the optimistic bias specific
  to contact or failure events?*

### D. Benchmark validity: use it as the instrument

The benchmark-audit paper (2606.04233) is valuable, but as a research
question "benchmarks are flawed" is a methodological challenge. Paper 3's
strongest negative result is that such clusters draw attention and resolve
nothing. Use its diagnostics (shortcut probe, paired significance,
within-distribution resampling, train-test proximity) as the measurement
standard for A, B and C. Prefer RoboCasa and RoboTwin 2.0, which pass more of
the diagnostics.

---

## 4. Recommended framing

**Working title:** *What does proprioception buy a generalist robot policy?
A controlled, pre-registered resolution of the state-shortcut contradiction
across spatial and embodiment shift.*

- **Framed as a contestable claim, not a methodological worry:** "State-free
  policies generalize better" against "EEF-relative state improves
  cross-embodiment transfer."
- **Explicit hypotheses:** H-shortcut, H-frame, H-observability, H-protocol,
  each with a kill criterion.
- **Resolving action, which is cheap and discriminating (Paper 4):** a
  factorial re-evaluation of *released* checkpoints and light fine-tunes, not
  new large-scale data collection.
- **Attention hook (Paper 3):** it plugs directly into the recognized
  cross-embodiment and VLA-design debates.

---

## 5. Further research to do before committing

Ordered roughly by cost. Steps 1–3 are cheap and can kill or confirm the
choice within a few weeks.

### 5.1 Build the evidence graph for the cluster (desk research, ~1 week)
- **Harvest.** Collect every 2019–2026 paper reporting a with/without-state
  or state-representation ablation, retrieving through the **citation graph**
  of 2509.18644, 2608.03052, 2609.02546 and De Haan 2019, not by keyword
  (Paper 4's lesson).
- **Extract per result:** backbone, action frame, state frame, history
  length, camera configuration and FOV, shift type (in-distribution, height,
  horizontal, embodiment strict or exposed), sim or real, number of trials,
  and effect size.
- **Load into the Paper 4 planner** (`scientific-evidence-planner`) with H1–H4
  and run gap analysis. Expect "no independent corroboration" gaps where
  results share a backbone, and "no evidence" gaps for H-observability.
- **Kill check.** If the extracted effects are already explained by one
  variable (for example, every "state hurts" result is absolute-frame and
  every "state helps" result is relative-frame), the question is resolved.
  Write the synthesis as a short paper and move to area C.

### 5.2 Protocol re-classification meta-analysis for area B (~1 week)
- Extend ZETA's Table 24 to every cross-embodiment "zero-shot" claim since
  RT-X.
- Test whether reported negative transfer clusters in one protocol class.
- If it does, the contradiction is a protocol artifact. That is a refutation
  result in Paper 3's sense, and publishable on its own.

### 5.3 Pilot the discriminating experiment (2–4 weeks, modest compute)
> **Superseded (2026-10-15):** the evidence map narrowed this to a 3 × 2 design (state encoding × shift type) on LIBERO-Plus. See `research/pilot_3x2_liberoplus.md`. The factorial below is kept for the record.

- **Factorial design:** state {none, absolute EEF, EEF-delta, noised} ×
  action {absolute, relative EEF} × cameras {wrist-only, wrist + third-person}
  × shift {in-distribution, height, horizontal, unseen embodiment}.
- **Setup:** an open π0/π0.5 (openpi/LeRobot) base, fine-tuned in RoboCasa
  and/or RoboTwin 2.0, with a small real-robot check on 1–2 tasks.
- **Statistics:** report per-episode outcomes and paired tests (2606.04233
  diagnostics). Pre-register the predicted sign pattern for each hypothesis
  before running anything.

### 5.4 Validate the Paper 3 predictors on robotics itself (this repo's natural job)
The rubric above assumes that astronomy-fitted predictors transfer to
robotics. That needs testing.

- **Backtest.** Freeze robotics questions at cutoffs such as 2021, 2022, 2023
  and 2024 from arXiv cs.RO plus cs.LG robot-learning papers. Use Paper 3's
  four sources: tension-mined, direct LLM, author future-work, and controls.
- **Labels.** Use 18–24-month horizons, since robotics moves faster than
  astronomy, and **citation-graph retrieval**, not TF-IDF.
- **Check.** See whether review overlap, evidence density, competing
  hypotheses and contradiction still predict uptake. Robotics has more
  industry labs and engineering claims, so the instrument-diversity and
  specificity effects may differ.
- **Payoff.** This tests the theory in a second domain (Paper 3's stated
  limitation: one domain) and is the most natural Paper 5 for this repository.

### 5.5 Keep area C warm
Using the public WMBench data, test a narrow hypothesis: *world-model
evaluators' success overestimation is concentrated in contact-sensitive and
failure transitions.* This is a cheap re-analysis with a quantified claim.
If 5.1 kills area A, this becomes the main line.

### 5.6 Measurement hygiene for all of the above
- Report strict and pretrain-exposed zero-shot results separately.
- Publish per-episode outcomes.
- Do not use LIBERO or SimplerEnv as the primary evidence.
- Record which provenance roots (backbone, dataset) each result shares, so
  later readers can apply Paper 4's independence logic.

---

## 6. Sources consulted

- Paper 3: `scientific-question-outcome-prediction-indicators/paper/paper3.md`
- Paper 4: `scientific-evidence-planner/paper/paper.tex`
- 2509.18644 Do You Need Proprioceptive States in Visuomotor Policies?
- 2608.03052 How Should Vision-Language-Action Models Use Proprioceptive State?
- 2609.02546 ZETA: Controlled Study of Zero-Shot Cross-Embodiment VLA Transfer
- 2608.18433 The Embodiment Gap in Robot Foundation Models (survey)
- 2606.04233 What Are We Actually Benchmarking in Robot Manipulation?
- 2607.02642 GigaWorld-1: World Models for Robot Policy Evaluation
- 2609.40341 Ego4WAM: Scaling Egocentric Human Data for Robot Learning
- Survey landscape: 2605.00080, 2606.00113, 2601.07823, 2607.24744, 2604.27621
