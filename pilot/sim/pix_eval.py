"""Closed-loop eval of pixel policies (pix_N / pix_R / pix_C). Images are rendered only at query steps
(sim.render at 128x128, no flip: matches the dataset convention, checked by pix_align.py), then INTER_AREA to 64x64
as in pix_extract.py. scene_blackout = agentview image all zeros from the first query (the BvF blackout).

usage: SEED=s pix_eval.py <suite> <model_dir> <out.jsonl> <conds> <inits> <arms> [task_ids]
"""
import libero_boot  # noqa
import collections, json, os, pathlib, re, sys, time
import numpy as np, torch, cv2
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
import eval_libero as ev
from layout_perturb import perturb_layout
from relative_state import EpisodeRelativeState
from pix_train import PixPolicy, to_float
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

suite_name, model_dir, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
conds = sys.argv[4].split(","); inits = ev._range(sys.argv[5]); arms = sys.argv[6].split(",")
task_ids = ev._range(sys.argv[7]) if len(sys.argv) > 7 else range(10)
torch.set_num_threads(1)
SEED = int(os.environ.get("SEED", "0")); SUF = "" if SEED == 0 else f"_s{SEED}"
RES = 64


def norm_task(s): return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


class Pol:
    def __init__(s, p):
        c = torch.load(p, weights_only=False); s.c = c
        s.net = PixPolicy(len(c["tasks"]), c["state_dim"], len(c["ymu"])); s.net.load_state_dict(c["state"]); s.net.eval()

    def act(s, av, wr, state, task):
        c = s.c; x = ((np.asarray(state, np.float64) - c["xmu"]) / c["xsd"]) if c["state_dim"] else np.zeros(0)
        oh = np.zeros(len(c["tasks"])); oh[c["tasks"].index(task)] = 1
        with torch.no_grad():
            y = s.net(to_float(av[None]), to_float(wr[None]), torch.tensor(oh[None], dtype=torch.float32),
                      torch.tensor(x[None], dtype=torch.float32))[0].numpy()
        return (y * c["ysd"] + c["ymu"]).reshape(c["H"], 7)


models = {a: Pol(f"{model_dir}/pix_{a}{SUF}.pt") for a in arms}
suite = benchmark.get_benchmark_dict()[suite_name]()
done_keys = set()
if out.exists():
    for l in out.read_text().splitlines():
        if l.strip(): r = json.loads(l); done_keys.add((r["arm"], r["task"], r["init"], r["condition"]))
t0 = time.time()
with out.open("a") as fh:
    for task_id in task_ids:
        task = suite.get_task(task_id); ntask = norm_task(task.language); inits_all = suite.get_task_init_states(task_id)
        env = ControlEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                         use_camera_obs=False, has_offscreen_renderer=True, has_renderer=False)
        env.seed(7)

        def cams(blackout):
            av = cv2.resize(env.env.sim.render(camera_name="agentview", width=128, height=128), (RES, RES), interpolation=cv2.INTER_AREA)
            wr = cv2.resize(env.env.sim.render(camera_name="robot0_eye_in_hand", width=128, height=128), (RES, RES), interpolation=cv2.INTER_AREA)
            return (np.zeros_like(av) if blackout else av), wr

        for init in inits:
            for cond in conds:
                inst = (suite_name, task_id, init)
                for arm in arms:
                    if (arm, task_id, init, cond) in done_keys: continue
                    ev._safe_reset(env); obs = env.set_init_state(inits_all[init]); pert = None
                    if cond.startswith("robot_init:"):
                        pert = ev.perturb_robot_init(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"))
                    if cond.startswith("layout:"):
                        obs, pert = perturb_layout(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"), ev.LIBERO_DUMMY_ACTION,
                                                   reset_fn=lambda: ev._safe_reset(env))
                        if obs is None:
                            fh.write(json.dumps(dict(arm=arm, seed=SEED, suite=suite_name, task=task_id, init=init, condition=cond,
                                                     skipped=True)) + "\n"); continue
                    for _ in range(10): obs, _, _, _ = env.step(ev.LIBERO_DUMMY_ACTION)
                    rel = EpisodeRelativeState(); plan = collections.deque(); done = False; t = 0
                    while t < ev.MAX_STEPS[suite_name]:
                        if not plan:
                            s = ev._state(obs); st = {"N": None, "R": s, "C": rel(s), "D": s}[arm]
                            av, wr = cams(cond == "scene_blackout")
                            plan.extend(models[arm].act(av, wr, st if st is not None else [], ntask)[:5])
                        obs, _, done, _ = env.step(np.clip(plan.popleft(), -1, 1).tolist()); t += 1
                        if done: break
                    fh.write(json.dumps(dict(arm=arm, seed=SEED, suite=suite_name, task=task_id, init=init, condition=cond,
                                             success=bool(done), steps=t, pert=pert), default=lambda o: np.asarray(o).tolist()) + "\n"); fh.flush()
        env.close()
        print(suite_name, "task", task_id, round(time.time() - t0), "s", flush=True)
os._exit(0)
