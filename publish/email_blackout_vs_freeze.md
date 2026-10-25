# Email to the Blackout vs Freeze authors

**To:** yohan.jo@snu.ac.kr (corresponding author)
**Cc:** heejaesuh@snu.ac.kr (first author)
**Subject:** Your π0.5 with/without-proprioception checkpoints could test a prediction about state harm under robot-start shifts

---

Dear Prof. Jo and Heejae Suh,

I enjoyed "Blackout vs. Freeze" (arXiv 2609.39145). In particular, the selective-removal result in Fig. 4 is close to an open question I have been working on: proprioception lifting π0.5 from 18.7% to 63.3% once the arm links and palm are removed from the scene view.

I am an independent researcher without GPU access. My question is when proprioceptive state helps or hurts a manipulation policy. I coded 123 with/without-state settings from 14 papers. In the literature, raw state hurt in all 16 configuration-shift settings, and helped in 11 of 14 camera-fault settings, which come from your paper.

I then ran a CPU-only study in LIBERO, with paired instances and 3 seeds. It used small policies trained with no state, raw state, or episode-relative state, at different perception levels. The main result:

- **The policy only sees gripper-relative object positions** (it cannot see the arm). Raw state hurts under a robot-start shift: −21 pp at a 0.1 rad L2 joint offset, the LIBERO-Plus generator.
- **The policy sees real 64×64 scene and wrist images.** The harm disappears (+4 pp, CI includes 0). The no-state policy sees the arm in the scene camera, so it is as sensitive to the start configuration as the state-using one (−25 vs −23 pp from in-distribution).

My reading is that "state hurts under robot-start shift" depends on whether the no-state policy can see the robot. My policies are small CNNs, though, not π0.5.

**Your with- and without-proprioception π0.5 checkpoints are exactly the pair that could test this at scale, with no new training.** Concretely:

1. **Evaluate both checkpoints under two shifts:**
   - robot-start 0.1: an L2 offset on the 7 joints in a seeded random direction;
   - object layout 6 cm: the target object moved, with the placement validated after settling.

   Use the same LIBERO initial states for both checkpoints, on Spatial and Object. My prediction is a with-minus-without difference within ±8 pp under the robot-start shift, and ≥ 0 under the layout shift.
2. **Optional: repeat the robot-start condition with the scene-view arm links removed**, using your renderer. Here my explanation predicts the opposite of your Fig. 4 result, so it is the most direct test: once the no-state policy cannot see the arm, state should matter, and the robot-start harm of raw state should reappear.

I have a paired evaluation client for openpi that implements both perturbations, with fixed seeds, validated placements and per-episode logs. Running it is a matter of pointing it at a policy server. At 150 episodes per cell, item 1 is about 1,200 episodes in total.

Draft write-up, code and per-episode results:
https://github.com/OpenFindings/scientific-discovery-robotics/tree/claude/friendly-archimedes-4ft6w0
(`research/paper_draft.md`; evaluation client in `pilot/stage0/eval_libero.py`)

If the predictions fail, I would report that just as prominently. I would of course credit your checkpoints and any runs. If you find the question interesting enough to collaborate, I would be glad to discuss co-authorship.

**Disclosure:** much of the literature coding, simulation code and analysis was done with an AI coding assistant (Claude) under my direction, and I have reviewed the results.

Thank you for considering this. A short reply, even "not possible", would help me plan.

Best regards,
[Your name]
[Website / contact]
