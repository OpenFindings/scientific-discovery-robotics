"""Offline tests for the Stage 0 client and analysis (no GPU, LIBERO or openpi needed).

LIBERO, robosuite and openpi_client are replaced by small fakes, so this
checks the logic the pilot depends on: the robot-init perturbation, the
fault timing, prompt logging, pairing determinism, and that the analysis
recovers known effects and applies the exit criteria.

  python -m pytest -q test_stage0_offline.py   (or: python test_stage0_offline.py)
"""
import json
import math
import pathlib
import sys
import tempfile
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --- stub the client-environment imports -------------------------------------------------
for name in ["libero", "libero.libero", "libero.libero.envs", "openpi_client", "tyro", "tqdm"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["libero.libero"].benchmark = types.SimpleNamespace()
sys.modules["libero.libero"].get_libero_path = lambda k: "/tmp"
sys.modules["libero.libero.envs"].OffScreenRenderEnv = object
sys.modules["tqdm"].tqdm = lambda x, **k: x
image_tools = types.SimpleNamespace(convert_to_uint8=lambda x: x.astype(np.uint8),
                                    resize_with_pad=lambda x, h, w: x[:h, :w])
sys.modules["openpi_client"].image_tools = image_tools
sys.modules["openpi_client"].websocket_client_policy = types.SimpleNamespace()
sys.modules["tyro"].cli = lambda f: None

import analyze_stage0 as an  # noqa: E402
import eval_libero as ev  # noqa: E402
import layout_perturb as lp  # noqa: E402
import relative_state as rs  # noqa: E402


# --- fakes ---------------------------------------------------------------------------------
class FakeModel:
    def __init__(self, names, ranges):
        self._ids = {n: i for i, n in enumerate(names)}
        self.jnt_range = np.array(ranges)

    def joint_name2id(self, n):
        return self._ids[n]


class FakeController:
    def __init__(self):
        self.initial_joints = None

    def update_initial_joints(self, q):
        self.initial_joints = np.array(q)


class FakeEnv:
    """7-joint arm at the LIBERO home pose; the gripper closes after `close_at` policy steps."""
    HOME = np.array([0, -0.161, 0.0, -2.445, 0.0, 2.227, np.pi / 4])

    def __init__(self, close_at=20, succeed_at=None):
        names = [f"j{i}" for i in range(7)]
        ranges = [[-2.9, 2.9], [-1.76, 1.76], [-2.9, 2.9], [-3.07, -0.07], [-2.9, 2.9], [-0.02, 3.75], [-2.9, 2.9]]
        qpos = np.concatenate([self.HOME, [0.04, -0.04]])
        self.sim = types.SimpleNamespace(data=types.SimpleNamespace(qpos=qpos, qvel=np.zeros(9)),
                                         model=FakeModel(names, ranges), forward=lambda: None)
        robot = types.SimpleNamespace(_ref_joint_pos_indexes=list(range(7)), _ref_joint_vel_indexes=list(range(7)),
                                      robot_joints=names, controller=FakeController())
        self.env = types.SimpleNamespace(robots=[robot], sim=self.sim)
        self.close_at, self.succeed_at, self.t = close_at, succeed_at, 0

    def reset(self):
        self.t = 0
        self.sim.data.qpos[:7] = self.HOME

    def set_init_state(self, s):
        return self._obs()

    def _obs(self):
        closed = self.t >= self.close_at + 10  # +10 settle steps
        g = 0.005 if closed else 0.04
        img = np.full((8, 8, 3), 100, np.uint8)
        return {"agentview_image": img, "robot0_eye_in_hand_image": img.copy(),
                "robot0_eef_pos": self.sim.data.qpos[:3].copy(), "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
                "robot0_gripper_qpos": np.array([g, -g])}

    def step(self, a):
        self.t += 1
        done = self.succeed_at is not None and self.t >= self.succeed_at + 10
        return self._obs(), 0.0, done, {}


class FakeClient:
    def __init__(self):
        self.calls = []

    def infer(self, el):
        self.calls.append(el)
        out = {"actions": np.zeros((10, 7))}
        if el.get("pilot/log_prompt"):
            out["pilot/prompt"] = "Task: x, State: 1 2 3;\nAction: "
        return out


ARGS = types.SimpleNamespace(resize_size=8, replan_steps=5, num_steps_wait=10, video_dir=None,
                             state_mode="absolute")


# --- client tests ----------------------------------------------------------------------------
def test_range_and_axisangle():
    assert ev._range("0-2,5") == [0, 1, 2, 5]
    assert np.allclose(ev._quat2axisangle([0, 0, 0, 1.0]), 0)
    s = np.sin(np.pi / 4)
    assert np.allclose(ev._quat2axisangle([s, 0, 0, np.cos(np.pi / 4)]), [np.pi / 2, 0, 0])


def test_robot_init_matches_liberoplus_parameterization():
    norms = []
    for k in range(50):
        env = FakeEnv()
        rec = ev.perturb_robot_init(env, 0.2, ev._rng("libero_spatial", 0, k, "robot_init:0.2", "dir"))
        norms.append(rec["applied_norm"])
        assert np.allclose(env.env.robots[0].controller.initial_joints, env.sim.data.qpos[:7])
        assert np.allclose(np.linalg.norm(rec["offset"]), rec["applied_norm"], atol=1e-4)
    # exact L2 norm unless a joint limit clipped it
    assert np.mean(np.isclose(norms, 0.2, atol=1e-4)) > 0.9


def test_pairing_is_deterministic_across_arms():
    a, b = FakeEnv(), FakeEnv()
    ra = ev.perturb_robot_init(a, 0.3, ev._rng("libero_goal", 3, 7, "robot_init:0.3", "dir"))
    rb = ev.perturb_robot_init(b, 0.3, ev._rng("libero_goal", 3, 7, "robot_init:0.3", "dir"))
    assert ra["offset"] == rb["offset"]


def test_scene_blackout_and_prompt_logging():
    env, cl = FakeEnv(close_at=10**6, succeed_at=30), FakeClient()
    rec, _ = ev.run_episode(env, cl, "task", None, "scene_blackout", ("s", 0, 0), ARGS, 220)
    assert rec["success"] and rec["steps"] == 30
    assert all(c["observation/image"].max() == 0 for c in cl.calls)
    assert all(c["observation/wrist_image"].max() > 0 for c in cl.calls)
    assert sum("pilot/log_prompt" in c for c in cl.calls) == 1 and "State:" in rec["prompt"]
    assert len(rec["states"]) == len(cl.calls) == 6


def test_wrist_blackout_starts_after_closure_plus_delay():
    env, cl = FakeEnv(close_at=20), FakeClient()
    rec, _ = ev.run_episode(env, cl, "task", None, "wrist_blackout_after_close", ("s", 1, 2), ARGS, 120)
    assert not rec["success"] and rec["closure_step"] is not None
    # gripper reads closed from policy step 20; three consecutive closed readings -> trigger at step 22
    assert rec["closure_step"] == 22
    assert rec["closure_step"] <= rec["fault_onset"] <= rec["closure_step"] + ev.MAX_DELAY
    for i, c in enumerate(cl.calls):  # queries happen at steps 0, 5, 10, ...
        assert (c["observation/wrist_image"].max() == 0) == (5 * i >= rec["fault_onset"])
        assert c["observation/image"].max() > 0


def test_id_condition_untouched():
    env, cl = FakeEnv(), FakeClient()
    rec, _ = ev.run_episode(env, cl, "task", None, "id", ("s", 0, 0), ARGS, 50)
    assert rec["perturbation"] is None and rec["fault_onset"] is None
    assert all(c["observation/image"].max() > 0 and c["observation/wrist_image"].max() > 0 for c in cl.calls)


# --- analysis tests --------------------------------------------------------------------------
def synth(effects, seed_sd=0.03, n_init=10, rng=None):
    """effects: {(arm, cond): success prob}; 4 suites x 10 tasks x n_init, two seeds per arm."""
    rng = rng or np.random.default_rng(1)
    rows = []
    for (arm, cond), p in effects.items():
        for seed in ("41", "42"):
            ps = np.clip(p + rng.normal(0, seed_sd), 0, 1)
            for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
                for task in range(10):
                    for i in range(n_init):
                        rows.append(dict(arm=arm, seed=seed, suite=suite, task=task, init=i, condition=cond,
                                         success=bool(rng.random() < ps),
                                         prompt="Task: t, State: 1;" if arm == "R" else "Task: t",
                                         states=[[0.0] * 8, [5.0] * 8] if cond.startswith("robot") else [[0.0] * 8]))
    return rows


GOOD = {("N", "id"): .94, ("R", "id"): .93, ("N", "scene_blackout"): .10, ("R", "scene_blackout"): .30,
        ("N", "robot_init:0.2"): .60, ("R", "robot_init:0.2"): .25}


def run_analysis(rows, extra=()):
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "x.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ns = pathlib.Path(d) / "ns.json"
        ns.write_text(json.dumps({"norm_stats": {"state": {"q01": [-1.0] * 8, "q99": [1.0] * 8,
                                                           "mean": [0] * 8, "std": [1] * 8}}}))
        out = pathlib.Path(d) / "s.json"
        extra = [x.replace("{ns}", str(ns)) for x in extra]
        sys.argv = ["analyze", str(p), "--norm-stats", str(ns), "--out", str(out), *extra]
        an.main()
        return json.loads(out.read_text())


def test_analysis_passes_on_expected_pattern():
    rep = run_analysis(synth(GOOD))
    c = rep["checks"]
    assert c["2_anchor"]["pass_"] and 10 <= c["2_anchor"]["R_minus_N"]["diff_pp"] <= 32
    lo, hi = c["2_anchor"]["R_minus_N"]["ci95_pp"]
    assert lo < c["2_anchor"]["R_minus_N"]["diff_pp"] < hi
    assert c["3_config_shift_bites"]["pass_"]
    assert c["prompt_flags"]["pass_"] and rep["stage0_exit"]
    assert rep["state_tokens"]["R|robot_init:0.2"]["share_tokens_saturated_255"] == 0.5


def test_analysis_fails_without_anchor_and_flags_prompt_mismatch():
    bad = dict(GOOD)
    bad[("R", "scene_blackout")] = .11
    rows = synth(bad)
    for r in rows:
        if r["arm"] == "N":
            r["prompt"] = "Task: t, State: 9;"  # N served with the state flag on
    rep = run_analysis(rows)
    assert not rep["checks"]["2_anchor"]["pass_"]
    assert not rep["checks"]["prompt_flags"]["pass_"] and not rep["stage0_exit"]


def test_seed_recommendation():
    assert an.recommend_seeds({"id": 1.0, "scene_blackout": 3.0})["seeds"] == 4
    assert an.recommend_seeds({"scene_blackout": 5.5})["seeds"] == 5
    assert an.recommend_seeds({"scene_blackout": 8.0})["seeds"] == "6-7"


# --- arm C: relative state -------------------------------------------------------------------
def _traj(rng, n=60, base_angle=3.1):
    """LIBERO-like states: EEF pointing down (rotation ~pi about x), drifting."""
    axis = np.array([1.0, 0.05, -0.02])
    rot = axis / np.linalg.norm(axis) * base_angle + np.cumsum(rng.normal(0, 0.01, (n, 3)), 0)
    pos = np.array([-0.05, 0.0, 0.68]) + np.cumsum(rng.normal(0, 0.005, (n, 3)), 0)
    grip = np.tile([0.039, -0.039], (n, 1))
    return np.concatenate([pos, rot, grip], 1)


def test_relative_roundtrip_and_anchor():
    rng = np.random.default_rng(0)
    for angle in (0.3, 2.9, 3.14159, 3.2, 4.0):  # LIBERO angles straddle pi
        s = _traj(rng, base_angle=angle)
        rel = rs.to_relative(s, s[0])
        assert np.abs(rel[0, :6]).max() < 1e-6 and np.allclose(rel[:, 6:], s[:, 6:])
        back = rs.from_relative(rel, s[0])
        assert np.abs(back[:, :3] - s[:, :3]).max() < 1e-5
        assert np.abs(rs.axis_angle_to_matrix(back[:, 3:6]) - rs.axis_angle_to_matrix(s[:, 3:6])).max() < 1e-5


def test_relative_rotation_ignores_global_frame():
    rng = np.random.default_rng(1)
    s = _traj(rng)
    G = rs.axis_angle_to_matrix(np.array([0.1, -0.2, 0.3]))
    moved = s.copy()
    moved[:, :3] += np.array([0.1, -0.05, 0.02])  # rigid translation of the whole episode
    moved[:, 3:6] = rs.matrix_to_axis_angle(np.einsum("ij,tjk->tik", G, rs.axis_angle_to_matrix(s[:, 3:6])))
    a, b = rs.to_relative(s, s[0]), rs.to_relative(moved, moved[0])
    assert np.allclose(a[:, :3], b[:, :3], atol=1e-6)    # translation removed
    assert np.allclose(a[:, 3:6], b[:, 3:6], atol=1e-5)  # start-orientation change removed


def test_online_encoder_matches_offline():
    rng = np.random.default_rng(2)
    s = _traj(rng)
    enc = rs.EpisodeRelativeState()
    online = np.stack([enc(x) for x in s])
    assert np.allclose(online, rs.to_relative(s, s[0]), atol=1e-6)
    enc.reset()
    assert np.abs(enc(s[5])[:6]).max() < 1e-6


def test_client_relative_mode_sends_zero_first_state():
    env, cl = FakeEnv(succeed_at=12), FakeClient()
    args = types.SimpleNamespace(**dict(vars(ARGS), state_mode="relative"))
    ev.perturb_robot_init(env, 0.3, ev._rng("s", 0, 0, "robot_init:0.3", "dir"))
    rec, _ = ev.run_episode(env, cl, "task", None, "id", ("s", 0, 0), args, 50)
    first = np.asarray(cl.calls[0]["observation/state"])
    assert np.abs(first[:6]).max() < 1e-6 and rec["state_mode"] == "relative"
    assert np.allclose(rec["states"][0][:6], 0)


def test_converter_on_synthetic_parquet():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("     (pyarrow missing; converter test skipped)")
        return
    import convert_relative_state as cv
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "src"
        (src / "meta").mkdir(parents=True)
        (src / "data" / "chunk-000").mkdir(parents=True)
        lengths = [30, 45]
        for ep, n in enumerate(lengths):
            s = _traj(rng, n=n).astype(np.float32)
            t = pa.table({"image": pa.array([{"bytes": b"png%d" % i, "path": None} for i in range(n)]),
                          "state": pa.FixedSizeListArray.from_arrays(pa.array(s.reshape(-1), pa.float32()), 8),
                          "actions": pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(n * 7, np.float32)), 7),
                          "frame_index": pa.array(np.arange(n), pa.int64()),
                          "episode_index": pa.array(np.full(n, ep), pa.int64())})
            t = t.replace_schema_metadata({b"huggingface": b"{}"})
            pq.write_table(t, src / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        (src / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.0", "total_episodes": 2}))
        (src / "meta" / "stats.json").write_text(json.dumps({"state": {}}))
        (src / "meta" / "episodes.jsonl").write_text(
            "".join(json.dumps({"episode_index": i, "length": n}) + "\n" for i, n in enumerate(lengths)))
        dst = pathlib.Path(d) / "dst"
        sys.argv = ["convert", "--src", str(src), "--dst", str(dst)]
        cv.main()
        for ep in range(2):
            a = pq.read_table(src / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
            b = pq.read_table(dst / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
            assert a.schema.equals(b.schema, check_metadata=True)
            assert all(a.column(c).equals(b.column(c)) for c in a.column_names if c != "state")
            st = np.array(b.column("state").to_pylist())
            assert np.abs(st[0, :6]).max() < 1e-6
            assert np.allclose(st, rs.to_relative(np.array(a.column("state").to_pylist()), a.column("state")[0].as_py()),
                               atol=1e-6)
        stats = json.loads((dst / "meta" / "stats.json").read_text())["state"]
        assert len(stats["mean"]) == 8
        assert json.loads((dst / "pilot_relstate_manifest.json").read_text())["frames"] == sum(lengths)


def test_check4_flags_out_of_range_C():
    rows = synth(GOOD)
    extra = []
    for r in synth({("C", "id"): .93, ("C", "scene_blackout"): .3, ("C", "robot_init:0.2"): .55}):
        r["states"] = [[0.0] * 8, [0.1] * 8]
        extra.append(r)
    rep = run_analysis(rows + extra, ["--norm-stats", "C=" + "{ns}"])
    c4 = rep["checks"]["4_C_in_distribution"]
    assert c4["pass_"] and c4["C_share_inside"]["robot_init:0.2"] == 1.0
    assert c4["R_share_inside_for_contrast"]["robot_init:0.2"] == 0.5


# --- anchor check ----------------------------------------------------------------------------
def _anchor_fixture(d, rng, n_demo=30, tasks=("pick the bowl", "open the drawer")):
    """Local LeRobot-v2-like dataset: per task, frame-0 anchors scattered ~10 mm in a plane."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = pathlib.Path(d) / "ds"
    (root / "meta").mkdir(parents=True)
    centers = {tasks[0]: np.array([-0.2, 0.0, 1.17]), tasks[1]: np.array([0.7, 0.0, 1.17])}  # ~0.9 m apart
    eps, idx = [], 0
    for task in tasks:
        for _ in range(n_demo):
            n = 20
            s = _traj(rng, n=n)
            s[:, :3] += centers[task] - s[0, :3] + np.r_[rng.normal(0, 0.006, 2), 0.0]
            f = root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"state": pa.FixedSizeListArray.from_arrays(
                pa.array(s.astype(np.float32).reshape(-1), pa.float32()), 8)}), f)
            eps.append({"episode_index": idx, "tasks": [task], "length": n})
            idx += 1
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.0", "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}))
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    return root, centers


def _eval_rows(rng, centers, shift_mm=(0, 0, 0), arms=("N", "R"), jitter_arm_mm=0.0, extra_task=None):
    rows = []
    for task, c in centers.items():
        for init in range(10):
            base = _traj(rng, n=1)[0]
            base[:3] = c + np.r_[rng.normal(0, 0.006, 2), 0.0] + np.array(shift_mm) / 1000
            for k, arm in enumerate(arms):
                st = base.copy()
                st[0] += k * jitter_arm_mm / 1000
                for cond in ("id", "scene_blackout", "robot_init:0.2"):
                    rows.append(dict(arm=arm, seed="41", suite="libero_spatial", task=list(centers).index(task),
                                     task_language=task.upper() + ".", init=init, condition=cond,
                                     perturbation={"applied_norm": 0.2} if cond.startswith("robot") else None,
                                     start_state=(st + (1.0 if cond.startswith("robot") else 0.0)).tolist()))
    if extra_task:
        rows.append(dict(arm="N", seed="41", suite="libero_goal", task=9, task_language=extra_task, init=0,
                         condition="id", perturbation=None, start_state=_traj(rng, n=1)[0].tolist()))
    return rows


def _run_anchor_check(d, root, rows):
    import check_anchors as ca
    p = pathlib.Path(d) / "eval.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = pathlib.Path(d) / "anchor.json"
    sys.argv = ["check_anchors", str(p), "--dataset", str(root), "--out", str(out)]
    try:
        ca.main()
    except SystemExit:
        pass
    return json.loads(out.read_text())


def test_anchor_check():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("     (pyarrow missing; anchor test skipped)")
        return
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as d:
        root, centers = _anchor_fixture(d, rng)
        good = _run_anchor_check(d, root, _eval_rows(rng, centers))
        assert good["pass_"], good["checks"]
        assert good["n_eval_instances"] == 20  # robot_init rows (start +1.0) were excluded
        assert max(abs(x) for x in good["systematic"]["mean_offset_mm"]) < 5
        shifted = _run_anchor_check(d, root, _eval_rows(rng, centers, shift_mm=(0, 0, 10)))
        assert not shifted["checks"]["systematic_pos"] and 8 < shifted["systematic"]["mean_offset_mm"][2] < 12
        unpaired = _run_anchor_check(d, root, _eval_rows(rng, centers, jitter_arm_mm=0.5))
        assert not unpaired["checks"]["pairing"]
        unmatched = _run_anchor_check(d, root, _eval_rows(rng, centers, extra_task="stack the blocks"))
        assert not unmatched["checks"]["all_tasks_matched"] and unmatched["unmatched_tasks"] == ["stack the blocks"]


# --- K-layout ---------------------------------------------------------------------------------
class LayoutFakeEnv(FakeEnv):
    """FakeEnv plus movable objects with toy physics: overlapping discs (r = 4 cm) push apart,
    objects past the table edge fall, and the goal is 'target within 2 cm of goal_xy'."""
    R, TABLE = 0.04, (-0.25, 0.25, -0.35, 0.35)

    def __init__(self, objects, interest, goal_xy=None, **kw):
        super().__init__(**kw)
        self.poses = {n: np.r_[xy, 0.90, 1.0, 0, 0, 0] for n, xy in objects.items()}
        objs = {n: types.SimpleNamespace(joints=[f"{n}_joint0"]) for n in objects}
        data = self.sim.data
        data.get_joint_qpos = lambda j: self.poses[j[:-len("_joint0")]].copy()
        data.set_joint_qpos = lambda j, q: self.poses.__setitem__(j[:-len("_joint0")], np.array(q, float))
        self.env.objects_dict, self.env.obj_of_interest = objs, interest
        self.goal_xy = goal_xy

    def get_sim_state(self):
        return dict(t=self.t, poses={n: p.copy() for n, p in self.poses.items()})

    def set_init_state(self, s):
        if s is not None:
            self.t, self.poses = s["t"], {n: p.copy() for n, p in s["poses"].items()}
        return self._obs()

    def check_success(self):
        tgt = self.env.obj_of_interest[0]
        return self.goal_xy is not None and np.linalg.norm(self.poses[tgt][:2] - self.goal_xy) < 0.02

    def step(self, a):
        names = list(self.poses)
        for i, n in enumerate(names):
            for m in names[i + 1:]:
                d = self.poses[m][:2] - self.poses[n][:2]
                dist = np.linalg.norm(d)
                if dist < 2 * self.R:
                    push = (2 * self.R - dist) / 2 * d / max(dist, 1e-9)
                    self.poses[n][:2] -= push
                    self.poses[m][:2] += push
            x, y = self.poses[n][:2]
            x0, x1, y0, y1 = self.TABLE
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                self.poses[n][2] -= 0.02
        return super().step(a)


def test_layout_moves_target_exactly_and_avoids_overlaps():
    retried = 0
    for k in range(20):
        env = LayoutFakeEnv({"bowl": [0.0, 0.0], "ramekin": [0.09, 0.0], "cookies": [-0.06, 0.07]},
                            interest=["bowl", "plate"])
        q0 = env.poses["bowl"].copy(), env.poses["ramekin"].copy()
        obs, rec = lp.perturb_layout(env, 0.06, ev._rng("s", 0, k, "layout:0.06", "dir"), ev.LIBERO_DUMMY_ACTION)
        assert obs is not None and rec["valid"] and rec["target"] == "bowl"
        # restored to the validated placement (settle undone), moved exactly 6 cm
        assert np.isclose(np.linalg.norm(env.poses["bowl"][:2] - q0[0][:2]), 0.06)
        assert np.allclose(env.poses["ramekin"], q0[1])
        for other in ("ramekin", "cookies"):
            # toy physics splits an overlap evenly, so a <= TOL_XY settle push allows <= 2*TOL_XY overlap
            assert np.linalg.norm(env.poses["bowl"][:2] - env.poses[other][:2]) >= 2 * env.R - 2 * lp.TOL_XY_M - 1e-9
        assert all(not t["ok"] for t in rec["tries"][:-1]) and rec["tries"][-1]["ok"]
        retried += rec["n_tries"] > 1
    assert retried > 0  # some first directions hit a neighbour and were re-drawn


def test_layout_rejects_overlap_falls_and_presolved():
    # target boxed in: every 6 cm move overlaps a neighbour or leaves the table
    ring = {f"o{i}": [0.20 + 0.06 * math.cos(i * math.pi / 4), 0.30 + 0.06 * math.sin(i * math.pi / 4)]
            for i in range(8)}
    env = LayoutFakeEnv(dict(bowl=[0.20, 0.30], **ring), interest=["bowl"])
    obs, rec = lp.perturb_layout(env, 0.06, ev._rng("s", 1, 0, "layout:0.06", "dir"), ev.LIBERO_DUMMY_ACTION)
    assert obs is None and not rec["valid"] and rec["n_tries"] == lp.MAX_TRIES
    assert np.allclose(env.poses["bowl"][:2], [0.20, 0.30])  # base state restored
    # every direction lands on the goal -> rejected as pre-solved
    env = LayoutFakeEnv({"bowl": [0.0, 0.0]}, interest=["bowl"], goal_xy=np.array([0.0, 0.0]))
    _, rec = lp.perturb_layout(env, 0.01, ev._rng("s", 2, 0, "layout:0.01", "dir"), ev.LIBERO_DUMMY_ACTION)
    assert not rec["valid"] and all(t["solved"] for t in rec["tries"])


def test_layout_paired_and_skips_fixture_only_tasks():
    a = LayoutFakeEnv({"bowl": [0.0, 0.0], "plate": [0.1, 0.1]}, interest=["bowl"])
    b = LayoutFakeEnv({"bowl": [0.0, 0.0], "plate": [0.1, 0.1]}, interest=["bowl"])
    _, ra = lp.perturb_layout(a, 0.03, ev._rng("libero_goal", 4, 2, "layout:0.03", "dir"), ev.LIBERO_DUMMY_ACTION)
    _, rb = lp.perturb_layout(b, 0.03, ev._rng("libero_goal", 4, 2, "layout:0.03", "dir"), ev.LIBERO_DUMMY_ACTION)
    assert ra["offset_m"] == rb["offset_m"] and ra["target"] == "bowl"
    # "turn on the stove": only a fixture of interest -> skipped, never run
    env, cl = LayoutFakeEnv({"bowl": [0.0, 0.0]}, interest=["flat_stove_1"]), FakeClient()
    rec, _ = ev.run_episode(env, cl, "turn on the stove", None, "layout:0.03", ("libero_goal", 3, 0), ARGS, 50)
    assert rec["skipped"] and rec["success"] is None and not cl.calls
    assert rec["perturbation"]["applicable"] is False


def test_analysis_drops_skipped_layout_rows():
    rows = synth({**GOOD, ("N", "layout:0.06"): .6, ("R", "layout:0.06"): .3})
    for arm in ("N", "R"):
        rows.append(dict(arm=arm, seed="41", suite="libero_goal", task=3, init=0, condition="layout:0.06",
                         skipped=True, success=None, prompt=None, states=[]))
    rep = run_analysis(rows)
    assert rep["skipped_layout_instances"] == {"N|layout:0.06": 1, "R|layout:0.06": 1}
    assert "layout:0.06" in rep["checks"]["3_config_shift_bites"]["R_minus_N"]
    assert rep["checks"]["3_config_shift_bites"]["R_minus_N"]["layout:0.06"]["diff_pp"] < -3


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(fails)
