"""Write the hand-extracted with/without-state results to data/results.jsonl.

Every number below was read from the cited table/figure of the paper
(location field). effect_pp = success(with state) - success(without
state), in percentage points, so NEGATIVE means state hurt.

comparison:
  presence  - same model trained with vs without proprioceptive input
  terms     - adding/removing specific state terms (velocities, joints,
              inter-gripper pose) on top of a base state
  frame     - absolute vs relative state encoding, both with state
  fusion    - same state, different fusion topology (both with state)
fusion: naive (concat / token / projector) or regularized (method that
  explicitly rebalances or grounds state: GAP, NADA, GeoProp, ...)
shift: ID (test inside training distribution), spatial (object / table /
  base position outside training range), embodiment, perturbation
  (state-vision mismatch injected at test time), robot_init (robot
  start configuration outside the demonstrated range), visual_fault
  (camera blackout / frozen frame / occlusion of the robot in the image;
  the robot configuration itself stays in distribution)
state_frame: absolute, relative (episode/EEF-relative), grounded (projected
  into the image), aligned (absolute but re-expressed in a frame consistent
  across embodiments, e.g. Mirage state alignment)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "results.jsonl"
rows = []


def add(paper, year, setting, effect, comparison="presence", **kw):
    base = dict(paper=paper, year=year, setting=setting, effect_pp=effect,
                comparison=comparison, fusion="naive", velocity_terms=False,
                state_history=1, masking_protocol="retrain")
    base.update(kw)
    rows.append(base)


# 2509.18644 Do You Need Proprioceptive States? (pi0 / ACT / DP, real)
c = dict(platform="real", backbone="pi0", pretrained=True, state_frame="absolute",
         action_space="relative_eef", wrist_camera=True)
add("2509.18644", 2025, "Pick Pen ID", 0.0, shift="ID", location="Table IX", **c)
add("2509.18644", 2025, "Pick Pen height", -98.4, shift="spatial", location="Table V", **c)
add("2509.18644", 2025, "Pick Pen horizontal", -58.4, shift="spatial", location="Table V", **c)
add("2509.18644", 2025, "Fold Shirt ID", 3.3, shift="ID", location="Table IX", **c)
add("2509.18644", 2025, "Fold Shirt horizontal", -65.1, shift="spatial", location="Table II", **c)
add("2509.18644", 2025, "Fetch Bottle whole-body horizontal", -66.7, shift="spatial", location="Table II", **c)
add("2509.18644", 2025, "Fold Shirt new embodiment, 5k FT steps", -36.7, shift="embodiment", location="Table VI", **c)
c2 = dict(c, pretrained=False)
add("2509.18644", 2025, "ACT Pick Pen height", -93.3, shift="spatial", location="Table V", **dict(c2, backbone="ACT"))
add("2509.18644", 2025, "ACT Pick Pen horizontal", -43.3, shift="spatial", location="Table V", **dict(c2, backbone="ACT"))
add("2509.18644", 2025, "DP Pick Pen height", -86.7, shift="spatial", location="Table V", **dict(c2, backbone="DiffusionPolicy"))
add("2509.18644", 2025, "DP Pick Pen horizontal", -53.3, shift="spatial", location="Table V", **dict(c2, backbone="DiffusionPolicy"))

# 2608.03052 How Should VLAs Use Proprioceptive State? (pi0.5, RoboCasa sim)
c = dict(platform="sim", benchmark="RoboCasa365", backbone="pi0.5", pretrained=True,
         state_frame="absolute", action_space="delta_eef", wrist_camera=True, shift="ID")
for name, eff in [("state prompt", 3.1), ("VLM prefix", 2.2), ("action prefix", 1.1),
                  ("state expert", 3.0), ("feature modulation", 3.0)]:
    add("2608.03052", 2026, f"atomic 45 tasks, {name}, K=1", eff, location="Table 1", **c)
add("2608.03052", 2026, "composite 20 tasks, VLM prefix, K=1", 6.0, location="Fig 4 / Table 2", **c)
add("2608.03052", 2026, "composite 20 tasks, action prefix, K=1", -0.2, location="Fig 4 / Table 2", **c)
add("2608.03052", 2026, "composite 20 tasks, action prefix, K=8", 10.6, location="Table 2",
    **dict(c, state_history=8))

# 2602.12032 When would Vision-Proprioception Policies Fail? (concat vs vision-only)
c = dict(backbone="ResNet18-transformer BC", pretrained=False, state_frame="absolute",
         action_space="delta_eef", state_history=5)
add("2602.12032", 2026, "MetaWorld+RoboSuite 8 tasks mean", -9.8, platform="sim", shift="ID",
    wrist_camera=False, location="Table 1", benchmark="MetaWorld/RoboSuite", **c)
add("2602.12032", 2026, "real 6 tasks mean", -20.0, platform="real", shift="ID",
    wrist_camera=True, location="Table 1", **c)
add("2602.12032", 2026, "sim OOD object position, 4 tasks", -15.3, platform="sim", shift="spatial",
    wrist_camera=False, location="Table 2", benchmark="MetaWorld/RoboSuite", **c)
add("2602.12032", 2026, "real OOD object position, 2 tasks", -20.0, platform="real", shift="spatial",
    wrist_camera=True, location="Table 2", **c)
add("2602.12032", 2026, "Octo fine-tuned VP vs V, 4 sim tasks", -10.3, platform="sim", shift="ID",
    wrist_camera=False, location="Table 3", benchmark="MetaWorld/RoboSuite",
    **dict(c, backbone="Octo", pretrained=True))
add("2602.12032", 2026, "GAP-regularized vs vision-only, 8 sim tasks", 8.1, platform="sim", shift="ID",
    fusion="regularized", wrist_camera=False, location="Table 1", benchmark="MetaWorld/RoboSuite", **c)
add("2602.12032", 2026, "GAP-regularized vs vision-only, sim OOD position", 11.0, platform="sim",
    shift="spatial", fusion="regularized", wrist_camera=False, location="Table 2",
    benchmark="MetaWorld/RoboSuite", **c)

# 2605.13067 When Absolute State Fails (ACT, real rail robot). Action space co-varies.
c = dict(platform="real", backbone="ACT", pretrained=False, wrist_camera=True,
         note="action space co-varies with state encoding")
add("2605.13067", 2026, "abs/abs vs no-state/chunk, ID", -15.0, shift="ID", state_frame="absolute",
    action_space="absolute_joint", location="Table II", **c)
add("2605.13067", 2026, "abs/abs vs no-state/chunk, OOD base position", -20.0, shift="spatial",
    state_frame="absolute", action_space="absolute_joint", location="Table II", **c)
add("2605.13067", 2026, "episode-relative vs no-state, ID", 30.0, shift="ID", state_frame="relative",
    action_space="relative_episode", location="Table II", **c)
add("2605.13067", 2026, "episode-relative vs no-state, OOD base position", 5.0, shift="spatial",
    state_frame="relative", action_space="relative_episode", location="Table II", **c)

# 2607.07101 GeoProp (Vanilla vs No-Proprio, and grounded vs No-Proprio). Object positions varied in-distribution.
for bb, pre, mw, rlb, rt in [("DP-ResNet18", False, -1.3, -2.0, 0.0), ("DP-ViT", False, 0.6, -5.4, -1.8)]:
    c = dict(platform="sim", backbone=bb, pretrained=pre, state_frame="absolute", wrist_camera=False, shift="ID")
    add("2607.07101", 2026, f"{bb} MetaWorld-50", mw, action_space="delta_eef", benchmark="MetaWorld", location="Table 1", **c)
    add("2607.07101", 2026, f"{bb} RLBench-6", rlb, action_space="absolute_eef", benchmark="RLBench", location="Table 1", **c)
    add("2607.07101", 2026, f"{bb} RoboTwin-7", rt, action_space="absolute_joint", benchmark="RoboTwin", location="Table 1", **c)
c = dict(state_frame="absolute", wrist_camera=False, shift="ID", action_space="absolute_joint")
add("2607.07101", 2026, "pi0 RoboTwin-7", 2.3, platform="sim", backbone="pi0", pretrained=True,
    benchmark="RoboTwin", location="Table 2", **c)
add("2607.07101", 2026, "DP real Mobile ALOHA 4 tasks", 10.0, platform="real", backbone="DP-ResNet18",
    pretrained=False, location="Table 4", **c)
add("2607.07101", 2026, "pi0 real Mobile ALOHA 4 tasks", 1.3, platform="real", backbone="pi0",
    pretrained=True, location="Table 4", **c)
for name, eff, plat, bb, pre in [("GeoProp DP-R18 sim overall", 7.2, "sim", "DP-ResNet18", False),
                                 ("GeoProp DP-ViT sim overall", 8.7, "sim", "DP-ViT", False),
                                 ("GeoProp pi0 RoboTwin", 6.3, "sim", "pi0", True),
                                 ("GeoProp DP real", 21.2, "real", "DP-ResNet18", False),
                                 ("GeoProp pi0 real", 11.3, "real", "pi0", True)]:
    add("2607.07101", 2026, name, eff, platform=plat, backbone=bb, pretrained=pre, fusion="regularized",
        state_frame="grounded", wrist_camera=False, shift="ID", action_space="mixed",
        location="Tables 1,2,4")

# 2601.16667 ReViP (pi0; state masked by zeroing - protocol unclear, possibly test-time)
c = dict(backbone="pi0", pretrained=True, state_frame="absolute", wrist_camera=True,
         action_space="delta_eef", masking_protocol="zeroed_unclear")
add("2601.16667", 2026, "LIBERO 4 suites", 3.7, platform="sim", benchmark="LIBERO", shift="ID",
    location="Supp. Table 12", **c)
add("2601.16667", 2026, "real cup pick-place, normal", 30.0, platform="real", shift="ID",
    location="Table 1", **c)
add("2601.16667", 2026, "real state-vision mismatch: recovery rate", -58.0, platform="real",
    shift="perturbation", location="Table 1", metric="back-to-object rate", **c)

# 2506.23944 Adapt Your Body (BC ResNet18+MLP, robomimic/MimicGen, delta EEF, wrist+3rd cam)
c = dict(platform="sim", backbone="ResNet18-MLP BC", pretrained=False, state_frame="absolute",
         action_space="delta_eef", wrist_camera=True, shift="ID", benchmark="robomimic/MimicGen")
full = dict(Square=-49.8, Can=-39.0, Coffee=2.5, HammerCleanup=-4.3, MugCleanup=11.9,
            Stack=-5.9, StackThree=-15.3, Threading=-11.0, ThreePieceAssembly=1.4)
for t, e in full.items():
    add("2506.23944", 2025, f"{t}: all state terms", e, velocity_terms=True, location="Table 1", **c)
pose = dict(Square=-15.5, Can=-3.1, MugCleanup=4.3, StackThree=-1.6, Threading=-0.8, ThreePieceAssembly=-2.8)
for t, e in pose.items():
    add("2506.23944", 2025, f"{t}: EE pose only", e, location="Table 5", **c)
nada = dict(Square=3.9, Can=3.0, Coffee=2.0, HammerCleanup=-1.1, MugCleanup=11.9,
            Stack=3.1, StackThree=7.7, Threading=3.8, ThreePieceAssembly=5.9)
for t, e in nada.items():
    add("2506.23944", 2025, f"{t}: NADA-regularized all terms", e, velocity_terms=True,
        fusion="regularized", location="Table 1", **c)

# 2108.03298 robomimic: adding EEF velocity / joint terms to EE-pose proprio (image BC-RNN)
c = dict(platform="sim", backbone="BC-RNN", pretrained=False, state_frame="absolute",
         action_space="delta_eef", wrist_camera=True, shift="ID", benchmark="robomimic",
         velocity_terms=True, location="Tables 22,25")
for t, ev, jt in [("Square PH", -17.3, -24.0), ("Transport PH", -7.3, -1.3),
                  ("Square MH", -30.0, -29.4), ("Transport MH", -32.0, -25.3)]:
    add("2108.03298", 2021, f"{t}: +EEF velocity", ev, comparison="terms", **c)
    add("2108.03298", 2021, f"{t}: +joint pos/vel", jt, comparison="terms", **c)

# 2405.12213 Octo: qualitative only (Appendix E)
add("2405.12213", 2024, "OXE pretraining + finetune, qualitative", None, platform="real", backbone="Octo",
    pretrained=True, state_frame="absolute", action_space="delta_eef", wrist_camera=True, shift="ID",
    direction=-1, location="Appendix E 'generally worse'")

# 2409.20537 HPT: from-scratch policy, real Sweep Leftover
add("2409.20537", 2024, "real Sweep Leftover, from scratch", 16.6, platform="real", backbone="HPT stem/head",
    pretrained=False, state_frame="absolute", action_space="mixed", wrist_camera=False, shift="ID",
    location="Table 3")

# 2402.10329 UMI: removing relative inter-gripper pose (bimanual), relative EE proprio kept
add("2402.10329", 2024, "bimanual cloth folding: inter-gripper relative pose", 40.0, comparison="terms",
    platform="real", backbone="DiffusionPolicy", pretrained=False, state_frame="relative",
    action_space="relative_chunk", wrist_camera=True, shift="ID", location="Fig 8")

# 2602.06575 ThinkProprio (FLOWER/Florence; CALVIN avg len /5 -> pp)
c = dict(platform="sim", backbone="FLOWER", pretrained=True, state_frame="absolute",
         action_space="delta_eef", wrist_camera=True, shift="ID", benchmark="CALVIN")
add("2602.06575", 2026, "MLP -> action head", 0.0, location="Table 7", **c)
add("2602.06575", 2026, "MLP -> VLM", -5.8, location="Table 7", **c)
add("2602.06575", 2026, "VLM-vocab tokens -> VLM", 0.8, location="Table 7", **c)

# 2609.02546 ZETA: absolute EEF state -> EEF-delta state, strict zero-shot embodiment shift
c = dict(comparison="frame", platform="sim", backbone="pi0.5-like VLA", pretrained=True,
         state_frame="relative", wrist_camera=True, shift="embodiment", benchmark="ZETA sim")
add("2609.02546", 2026, "world-delta action: abs -> EEF-delta state, avg 7 targets", 13.1,
    action_space="delta_eef", location="Table 2", **c)
add("2609.02546", 2026, "EEF-delta action: abs -> EEF-delta state, avg 7 targets", 11.1,
    action_space="relative_eef", location="Table 2", **c)

# 2610.02813 RRDF: dense vs delayed fusion, both with state (ACT)
c = dict(comparison="fusion", backbone="ACT", pretrained=False, state_frame="absolute",
         action_space="absolute_joint", wrist_camera=True, fusion="regularized")
add("2610.02813", 2026, "robomimic Square nominal", 18.0, platform="sim", shift="ID", location="Table I", **c)
add("2610.02813", 2026, "real held-out positions, 3 tasks mean", 51.7, platform="real", shift="spatial",
    location="Table II", **c)

# 2609.39145 Blackout vs Freeze: pi0.5 and GR00T retrained with vs without proprio,
# LIBERO Spatial/Object/Goal, n=450 per condition; EEF+gripper state, absolute.
for bb, vals in [("pi0.5", dict(clean=-1.3, scene_blackout=19.8, wrist_blackout=11.3,
                                 scene_frozen=5.6, wrist_frozen=3.6)),
                 ("GR00T", dict(clean=1.6, scene_blackout=17.2, wrist_blackout=12.2,
                                 scene_frozen=4.2, wrist_frozen=0.0))]:
    c = dict(platform="sim", benchmark="LIBERO", backbone=bb, pretrained=True, state_frame="absolute",
             action_space="delta_eef", wrist_camera=True, location="Table 5")
    for k, e in vals.items():
        add("2609.39145", 2026, f"{bb} {k.replace('_', ' ')}", e,
            shift="ID" if k == "clean" else "visual_fault", **c)
for bb, arm, wr, obj in [("pi0.5", 44.6, 42.6, 0.0), ("GR00T", 30.0, 68.6, 0.0)]:
    c = dict(platform="sim", benchmark="LIBERO", backbone=bb, pretrained=True, state_frame="absolute",
             action_space="delta_eef", wrist_camera=True, shift="visual_fault", location="Table 10")
    add("2609.39145", 2026, f"{bb} scene view: arm links + palm masked", arm, **c)
    add("2609.39145", 2026, f"{bb} wrist view: robot masked", wr, **c)
    add("2609.39145", 2026, f"{bb} wrist view: objects masked", obj, **c)

# 2511.00555 D3P: Diffusion Policy vision+joint proprio vs vision-only, RLBench 6 tasks x 40
c = dict(platform="sim", benchmark="RLBench", backbone="DiffusionPolicy", pretrained=False,
         state_frame="absolute", action_space="absolute_joint", wrist_camera=True, location="Table I")
add("2511.00555", 2025, "6 tasks, fixed initial joints", 3.4, shift="ID", **c)
add("2511.00555", 2025, "6 tasks, initial joints randomized +-10 deg", -22.5, shift="robot_init", **c)

# 2512.20166 LoLA: Qwen2.5-VL + flow-matching expert, with vs without state (state injected
# through its SALR state transformer; no anti-shortcut mechanism, so coded naive). LIBERO.
c = dict(platform="sim", benchmark="LIBERO", backbone="Qwen2.5-VL LoLA", pretrained=True,
         state_frame="absolute", action_space="delta_eef", wrist_camera=True, shift="ID", location="Table 7")
for suite, e in [("Goal", 7.2), ("Object", 6.0), ("Spatial", 2.5), ("Long", 10.5)]:
    add("2512.20166", 2025, f"LIBERO-{suite}", e, **c)

# 2610.07015 LEAP: state kept out of vs also fed to the auxiliary geometry decoder (both
# policies use state in the action head). effect = keep-out minus also-in. RoboTwin, 5 tasks x 300.
c = dict(comparison="fusion", platform="sim", benchmark="RoboTwin", pretrained=False,
         state_frame="absolute", action_space="absolute_joint", wrist_camera=True, shift="ID",
         fusion="regularized", location="Tables 3, 6")
add("2610.07015", 2026, "DP: state out of geometry decoder", 18.4, backbone="DiffusionPolicy", **c)
add("2610.07015", 2026, "ACT: state out of geometry decoder", 12.2, backbone="ACT", **c)

# 2610.14854 DEFT-VLA: state-token masking curriculum vs none (both with state), real pipette task, n=20
add("2610.14854", 2026, "pipette tip attachment: state-token masking", 55.0, comparison="fusion",
    platform="real", backbone="pi0.5", pretrained=True, state_frame="absolute", action_space="delta_eef",
    wrist_camera=True, shift="ID", fusion="regularized", location="Table III")

# 2402.19249 Mirage: separate state+image vs image-only policies, same data/architecture.
# Sim: robosuite BC-RNN, EEF pose state (robosuite aligns EEF frames across arms, so the
# target-robot state is frame-aligned); 5 tasks, means over Lift/Stack/Can/TwoPiece/Square.
c = dict(platform="sim", benchmark="robosuite", backbone="BC-RNN", pretrained=False,
         state_frame="aligned", action_space="delta_eef", wrist_camera=False, location="Tables II, III")
for t, e in dict(Lift=-2, Stack=0, Can=34, TwoPieceAssembly=9, Square=-7).items():
    add("2402.19249", 2024, f"source Franka {t}", float(e), shift="ID", **c)
for name, e in [("UR5e, cross-painted, Franka gripper", 18.4), ("UR5e, cross-painted, default gripper", 25.6),
                ("Kinova, cross-painted, Franka gripper", 4.0), ("Kinova, cross-painted, default gripper", 11.2)]:
    add("2402.19249", 2024, f"{name}, 5-task mean", e, shift="embodiment", **c)
for name, e in [("UR5e, raw images, Franka gripper", -7.8), ("UR5e, raw images, default gripper", -7.2),
                ("Kinova, raw images, Franka gripper", -5.0), ("Kinova, raw images, default gripper", 2.2)]:
    add("2402.19249", 2024, f"{name}, 5-task mean", e, shift="embodiment", **c)
# Real: Diffusion Policy, third-person camera only, 4 tasks (Tiger/Drawer/Cup/Toaster), 10% steps
c = dict(platform="real", backbone="DiffusionPolicy", pretrained=False, action_space="delta_eef",
         wrist_camera=False, location="Table IV")
add("2402.19249", 2024, "real source robot, 4-task mean", 7.5, shift="ID", state_frame="absolute", **c)
add("2402.19249", 2024, "real target gripper, state aligned + cross-painting", 5.0, shift="embodiment",
    state_frame="aligned", **c)
add("2402.19249", 2024, "real UR5, state aligned + cross-painting", 12.5, shift="embodiment",
    state_frame="aligned", **c)
add("2402.19249", 2024, "real target gripper, state NOT aligned + cross-painting", -40.0, shift="embodiment",
    state_frame="absolute", **c)
add("2402.19249", 2024, "real UR5, state NOT aligned + cross-painting", -45.0, shift="embodiment",
    state_frame="absolute", **c)

# 2508.08113 AimBot: pi0 trained with zeroed proprio vs with proprio, LIBERO, 4 runs/suite
c = dict(platform="sim", benchmark="LIBERO", backbone="pi0", pretrained=True, state_frame="absolute",
         action_space="delta_eef", wrist_camera=True, shift="ID", location="Tables 4, 8")
for suite, e, e_aim in [("Spatial", 0.2, 0.3), ("Object", 2.4, 1.6), ("Goal", 1.0, 2.4), ("Long", 2.0, 3.0)]:
    add("2508.08113", 2025, f"pi0 LIBERO-{suite}", e, **c)
    add("2508.08113", 2025, f"pi0+AimBot overlay LIBERO-{suite}", e_aim, **c)

# NEW_ROWS_MARKER

OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "records ->", OUT)
