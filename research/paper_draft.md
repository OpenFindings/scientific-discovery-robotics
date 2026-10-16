# When does proprioception hurt a manipulation policy? The shift type sets the sign

*Draft. A literature meta-analysis plus a CPU simulation study, with testable predictions for large VLA policies. No GPU was used: the simulation policies are small networks trained on CPU. The predictions for π0.5-class models are stated so that a group with GPUs can confirm or refute them.*

<!-- SECTIONS 0 (abstract) and 4 (pixel results) are filled after the CPU runs finish -->

## 1. The contradiction

Robot learning papers disagree about the robot's own joint and gripper readings ("proprioceptive state"):

- Some find that removing state makes policies more robust, by tens of points. Example: "state-free" policies under height and horizontal shifts (2509.18644).
- Others find that state rescues a policy when the camera fails, by up to +69 points (Blackout vs Freeze, 2609.39145).
- Others find it adds a few points in distribution, or nothing.

The usual explanation is a **shortcut**. State predicts the next action so well during training that the policy leans on it and stops looking. That explains the harms, not the rescues.

We ask which variable decides the sign.

## 2. Meta-analysis (literature)

**Retrieval.** We walked the citation graph, both directions and two hops, from the three conflicting papers and the two papers that introduced the shortcut idea. That gave 912 candidates. A high-recall screen flagged 89, and all of them were read. **14 papers report 123 with-vs-without-state settings.** All are coded in `evidence_map/data/results.jsonl` with source tables and figures.

**Pre-set rule.** A variable *explains* the contradiction if:
- it predicts the sign of the effect with paper-weighted accuracy of at least 0.90; and
- no level of it has counterexamples from two or more papers.

**Result.** No single variable passes. The best is the type of evaluation shift, at 0.78. But every **shifted** cell is now nearly clean:

| Cell | Settings | Outcome |
|---|---|---|
| Configuration shift (object, base, embodiment or start-joint out of range), raw state | 16 settings, 6 papers | **hurt 16/16** |
| Configuration shift, corrected frame (relative / grounded / aligned) or regularized fusion | 12 settings, 3 papers | help 8, hurt 3 (all three with broken images), null 1 |
| Camera fault (blackout, frozen frame, robot masked), robot in distribution | 14 settings, 1 paper | **help 11/14**, no hurt |
| In distribution, raw state | 60 settings, 14 papers | help 15, hurt 14, null 31; mostly within ±3 pp |

**Planner.** Seven hypotheses were loaded into a Bayesian evidence planner (Paper 4's framework). The posterior favours **H7, shift type × encoding** (0.67), over H6, shift type alone (0.28), and the original "any shift hurts" (0.05). H6 and H7 were formulated after reading, so this ranks explanations and does not confirm them.

The planner's most informative missing cells are all *configuration shift with a corrected encoding*. No paper tests that encoding under both a configuration shift and a camera fault.

**What the meta-analysis cannot separate.** Its "configuration shift" pools two kinds of shift:
- the robot itself starting somewhere new;
- the objects being somewhere new.

Raw state hurt under both in the published rows. But published object-position harms come from policies whose visual grounding is unknown.

## 3. Simulation study (CPU, real LIBERO)

We ran the shifts in real LIBERO (MuJoCo 3.2.3, robosuite 1.4.1) on paired instances: every policy sees identical starts and perturbations. The shifts were:

- **robot-init:** an L2 offset of 0.1 rad on the 7 joints, as in the LIBERO-Plus generator;
- **layout:** the object of interest moved 6 cm, with validated placement;
- **scene blackout:** the agent-view image is zero.

Policies predict 10-step action chunks and execute 5. All comparisons are paired, with 95% task-cluster bootstrap CIs.

Every perception level was trained with the same three state arms:
- **N:** no state;
- **R:** raw absolute end-effector state;
- **C:** episode-relative state (pose relative to the first frame; gripper absolute).

| Study | Perception | Suites | Seeds |
|---|---|---|---|
| E2 blind | none (task id only) | all 4 | 3 |
| E3 allocentric oracle | world positions of the objects | Spatial, Object | 1 |
| E4 ego oracle | object minus gripper position (a perfect wrist camera) | Spatial, Object | 3 |
| E5 pixels | 64×64 agent-view + wrist images, small CNN | Spatial, Object | 3 |

<!-- tables filled from pilot/sim/results -->

## 5. Testable predictions for large VLAs

<!-- filled after E5 -->

## 6. Limitations

<!-- filled -->
