# Novelty check: blind-solvable LIBERO and the robot-init vs layout sign reversal

Date: 2026-10-16. Tool: alphaXiv search plus full-text reads. Scope: arXiv up to mid-October 2026.

## Claims checked

| # | Our claim (from `sim_findings_cpu_libero.md`) | Closest prior work | Verdict |
|---|---|---|---|
| C1 | A policy with **no images at all** (absolute proprioception + task id, small MLP) solves ~72% of in-distribution LIBERO | LIBERO-PRO (2510.03827): trajectories are unchanged when the object is removed or the instruction is corrupted; MINERVA (2609.03715): a 0.54M CNN gets 95%, and permuting the task id drops it to chance; 2610.06078: VLAs replay behaviours | **Known in spirit; the exact number is new.** "LIBERO is memorization" is established. No paper we found reports a fully image-free LIBERO baseline. We present the 72% as a sharper quantification of LIBERO-PRO and MINERVA, not as a discovery. |
| C2 | The effect of state **reverses sign** with the shift type: state hurts under a robot-init shift (ego oracle R−N −20 pp) and helps under an object-layout shift (C−N +18 pp at 6 cm) | LIBERO-Plus and MINERVA report robot-init and layout degradation separately, without a state ablation. LIBERO-MAX (2609.36518): mid-task relocation, no state ablation. 2609.09213: a 5 cm object displacement collapses all visual policies, with or without extra geometry, and no robot-init condition | **Not found.** No paper crosses state on/off with robot-init vs layout shift. |
| C3 | An episode-relative state frame does **not** remove the robot-init harm once the policy has perception (ego: C−R −2 pp). It helps only for a blind policy (+20 pp). | 2605.13067: episode-relative state is best for a moving-frame (rail) start shift; there absolute/absolute fails. 2609.09213: a relative-coordinate state-only policy is invariant to frame relabelling (7/10 at all angles). | **Partly new.** Prior positive results use state-only or moving-frame setups, which match our blind-regime result. The failure of the relative frame with perception under a joint-space start offset is not reported. |
| C4 | Harm from state under layout shift in real VLAs is best read as vision suppression, not as a frame mismatch | Blackout-vs-Freeze and the evidence-map papers (vision suppression); 2608.11769 (initial-pose shortcut on one humanoid task, modulated by the wrist camera) | **Interpretation, consistent with prior work.** It is a hypothesis to test, not a finding. |

## Consequences for the story

1. Do not lead with "LIBERO can be solved blind". Lead with the **shift-type × state interaction** (C2) and the **limits of the relative frame** (C3). Use C1 as a calibration floor: the blind floor shows how much success is attainable without perception, so success rates under shift should be read against it.
2. Cite LIBERO-PRO, MINERVA and 2610.06078 for memorization; cite 2605.13067 and 2609.09213 as the prior evidence for the relative frame, and say how our regime differs (a perceiving policy with a joint-space start offset).
3. The oracle experiments are the weak point. A reviewer can say the "vision" is perfect object coordinates. The pixel-CNN replication (task #2) is what turns C2/C3 from an oracle result into a policy result.

## Not checked or not accessible

- "Dismantling the illusion…" (Zhou, ICML 2026): abstract only, on memorization. No state × shift analysis appears in it.
- LIBERO-X: perturbation hierarchy, no state ablation according to the LIBERO-MAX related-work section.
- Non-arXiv venues (CoRL/RSS workshop papers without a preprint) were not searched.
