"""Stage 0 LIBERO evaluation client: paired instances x conditions, one JSON line per episode.

Runs in openpi's LIBERO client environment (Python 3.8, examples/libero:
robosuite 1.4, openpi-client), against a policy served by serve.py.
Preprocessing, settling and episode limits follow openpi's
examples/libero/main.py, which is also what Blackout vs Freeze used.

Conditions (comma-separated --conditions):
  id                         unperturbed
  scene_blackout             agent-view image all zeros from the first policy step
  wrist_blackout_after_close wrist image all zeros from closure + U{0..10} steps
                             (closure: finger width < 0.020 m for 3 consecutive steps;
                             Blackout vs Freeze App. A.2)
  robot_init:<m>             arm joints offset by m * u at the start, u a uniformly random
                             unit vector in 7-D joint space (radians). This reproduces
                             LIBERO-Plus's robot-init perturbation, whose 500 released
                             offsets have L2 norms of exactly 0.1-0.5.

State encoding (--state-mode): `absolute` sends LIBERO's 8-D state as is (arms N, R);
`relative` sends relative_state.to_relative(state, first state of the episode)
(arm C), the same function used to build its training set. The anchor is the
state at the first policy query, after the settle steps.

Every random draw (perturbation direction, fault delay) is seeded from the
instance and condition only, so all arms and seeds see identical instances.

  python eval_libero.py --arm R --seed-label 41 --port 8000 \
      --conditions id,scene_blackout,robot_init:0.2 --out results/stage0/R_s41.jsonl
"""
import collections
import dataclasses
import json
import logging
import math
import pathlib
import time
import zlib
from typing import List, Optional

import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from relative_state import EpisodeRelativeState

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
CLOSE_WIDTH, CLOSE_STEPS, MAX_DELAY = 0.020, 3, 10


@dataclasses.dataclass
class Args:
    arm: str  # N | R | C | released | base_probe (label only; the server decides behaviour)
    seed_label: str  # training seed of the served checkpoint (label only)
    out: str  # JSONL output path; existing episodes are skipped (resume)
    host: str = "127.0.0.1"
    port: int = 8000
    suites: str = "libero_spatial,libero_object,libero_goal,libero_10"
    tasks: str = "0-9"
    init_states: str = "0-9"
    conditions: str = "id,scene_blackout,robot_init:0.2"
    state_mode: str = "absolute"  # absolute (N, R) | relative (C)
    eval_seed: int = 7
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    video_dir: Optional[str] = None


def _range(spec: str) -> List[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("/".join(map(str, key)).encode()))


def _quat2axisangle(quat):
    """Same as openpi examples/libero/main.py (copied from robosuite)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _state(obs) -> np.ndarray:
    return np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]))


def _finger_width(obs) -> float:
    q = obs["robot0_gripper_qpos"]
    return float(abs(q[0]) + abs(q[1]))


def perturb_robot_init(env, magnitude: float, rng: np.random.Generator) -> dict:
    """Offset the 7 arm joints by magnitude * (random unit vector), within joint limits.

    Also moves the OSC controller's nullspace target to the new configuration,
    as LIBERO-Plus does implicitly by changing the robot's init_qpos.
    """
    renv = env.env
    robot = renv.robots[0]
    sim = renv.sim
    idx = list(robot._ref_joint_pos_indexes)
    q0 = sim.data.qpos[idx].copy()
    u = rng.normal(size=len(idx))
    u /= np.linalg.norm(u)
    lo = np.array([sim.model.jnt_range[sim.model.joint_name2id(j)][0] for j in robot.robot_joints])
    hi = np.array([sim.model.jnt_range[sim.model.joint_name2id(j)][1] for j in robot.robot_joints])
    q1 = np.clip(q0 + magnitude * u, lo + 1e-3, hi - 1e-3)
    sim.data.qpos[idx] = q1
    sim.data.qvel[list(robot._ref_joint_vel_indexes)] = 0.0
    sim.forward()
    if hasattr(robot.controller, "update_initial_joints"):
        robot.controller.update_initial_joints(q1)
    return dict(requested_norm=magnitude, applied_norm=float(np.linalg.norm(q1 - q0)),
                clipped=bool(np.any(q1 != q0 + magnitude * u)), offset=(q1 - q0).round(5).tolist())


def _preprocess(img, size):
    img = np.ascontiguousarray(img[::-1, ::-1])  # rotate 180 deg to match training data
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


def run_episode(env, client, task_description, init_state, cond, inst, args, max_steps):
    env.reset()
    obs = env.set_init_state(init_state)
    rec = dict(perturbation=None, fault_onset=None, closure_step=None)
    if cond.startswith("robot_init:"):
        rec["perturbation"] = perturb_robot_init(env, float(cond.split(":")[1]), _rng(*inst, cond, "dir"))
    delay = int(_rng(*inst, cond, "delay").integers(0, MAX_DELAY + 1))
    if cond == "scene_blackout":
        rec["fault_onset"] = 0

    action_plan = collections.deque()
    encode = EpisodeRelativeState() if args.state_mode == "relative" else (lambda x: x)
    states, frames, prompt = [], [], None
    closed_run, done, t = 0, False, 0
    for _ in range(args.num_steps_wait):  # let objects settle; the arm holds its (possibly perturbed) pose
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    start_state = _state(obs)
    while t < max_steps:
        img = _preprocess(obs["agentview_image"], args.resize_size)
        wrist = _preprocess(obs["robot0_eye_in_hand_image"], args.resize_size)
        if cond == "scene_blackout":
            img = np.zeros_like(img)
        if cond == "wrist_blackout_after_close" and rec["fault_onset"] is not None and t >= rec["fault_onset"]:
            wrist = np.zeros_like(wrist)
        if args.video_dir:
            frames.append(np.concatenate([img, wrist], axis=1))
        if not action_plan:
            state = np.asarray(encode(_state(obs)), dtype=np.float32)  # what the policy is sent
            states.append(np.round(state, 5).tolist())
            element = {"observation/image": img, "observation/wrist_image": wrist,
                       "observation/state": state, "prompt": str(task_description)}
            if prompt is None:
                element["pilot/log_prompt"] = True
            result = client.infer(element)
            if prompt is None:
                prompt = result.get("pilot/prompt", "<server did not log prompt>")
            action_plan.extend(result["actions"][: args.replan_steps])
        obs, _, done, _ = env.step(action_plan.popleft().tolist())
        t += 1
        if done:
            break
        closed_run = closed_run + 1 if _finger_width(obs) < CLOSE_WIDTH else 0
        if (cond == "wrist_blackout_after_close" and rec["closure_step"] is None
                and closed_run >= CLOSE_STEPS):
            rec["closure_step"] = t
            rec["fault_onset"] = t + delay
    rec.update(success=bool(done), steps=t, prompt=prompt, start_state=start_state.round(5).tolist(),
               states=states, state_mode=args.state_mode)
    return rec, frames


def main(args: Args) -> None:
    if args.state_mode not in ("absolute", "relative"):
        raise ValueError(f"--state-mode must be absolute or relative, not {args.state_mode}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done_keys.add((r["suite"], r["task"], r["init"], r["condition"]))
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_meta = client.get_server_metadata().get("pilot", {})
    logging.info("server: %s", server_meta)

    bench = benchmark.get_benchmark_dict()
    with out.open("a") as fh:
        for suite_name in args.suites.split(","):
            suite = bench[suite_name]()
            for task_id in tqdm.tqdm(_range(args.tasks), desc=suite_name):
                task = suite.get_task(task_id)
                inits = suite.get_task_init_states(task_id)
                bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                                         camera_widths=LIBERO_ENV_RESOLUTION)
                env.seed(args.eval_seed)
                for init_idx in _range(args.init_states):
                    for cond in conditions:
                        key = (suite_name, task_id, init_idx, cond)
                        if key in done_keys:
                            continue
                        t0 = time.time()
                        try:
                            rec, frames = run_episode(env, client, task.language, inits[init_idx], cond,
                                                      (suite_name, task_id, init_idx), args,
                                                      MAX_STEPS[suite_name])
                        except Exception as e:  # counted as a failure, as openpi's main.py does
                            logging.exception("episode %s failed", key)
                            rec, frames = dict(success=False, error=repr(e)), []
                        row = dict(arm=args.arm, seed=args.seed_label, suite=suite_name, task=task_id,
                                   task_language=task.language, init=init_idx, condition=cond,
                                   eval_seed=args.eval_seed, server=server_meta,
                                   wall_s=round(time.time() - t0, 1), **rec)
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        if args.video_dir and frames:
                            import imageio
                            vdir = pathlib.Path(args.video_dir)
                            vdir.mkdir(parents=True, exist_ok=True)
                            name = f"{args.arm}_s{args.seed_label}_{suite_name}_t{task_id}_i{init_idx}_{cond}.mp4"
                            imageio.mimwrite(vdir / name.replace(":", "-"), frames, fps=10)
                env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
