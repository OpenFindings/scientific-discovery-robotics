"""Closed-loop eval of blind policies (T / A / Rl) in LIBERO, no rendering, with the pilot's
own perturbations. Paired: every policy sees the identical instance + perturbation."""
import libero_boot  # noqa: F401
import collections, json, pathlib, re, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
import eval_libero as ev
from layout_perturb import perturb_layout
from relative_state import EpisodeRelativeState
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

suite_name, model_dir, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
conds = sys.argv[4].split(",")
inits = ev._range(sys.argv[5]) if len(sys.argv) > 5 else list(range(5))
policies = sys.argv[6].split(",") if len(sys.argv) > 6 else ["T", "A", "Rl"]
torch.set_num_threads(1)
import os
SEED = int(os.environ.get('SEED', '0'))
SUF = '' if SEED == 0 else f'_s{SEED}'


def norm_task(s):
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


class Blind:
    def __init__(self, path):
        ck = torch.load(path, weights_only=False)  # our own checkpoint, written by blind_train.py
        self.c = ck
        n_in = ck["in_dim"] + len(ck["tasks"])
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                                       torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(),
                                       torch.nn.Linear(512, len(ck["ymu"])))
        self.net.load_state_dict(ck["state"]); self.net.eval()

    def act(self, feat, task):
        ck = self.c
        x = (np.asarray(feat, np.float64) - ck["xmu"]) / ck["xsd"]
        oh = np.zeros(len(ck["tasks"])); oh[ck["tasks"].index(task)] = 1
        with torch.no_grad():
            y = self.net(torch.tensor(np.r_[x, oh], dtype=torch.float32)[None])[0].numpy()
        return (y * ck["ysd"] + ck["ymu"]).reshape(ck["H"], 7)


models = {p: Blind(f"{model_dir}/blind_{p}{SUF}.pt") for p in policies}
suite = benchmark.get_benchmark_dict()[suite_name]()
t0 = time.time()
done_keys = set()
if out.exists():
    for l in out.read_text().splitlines():
        if l.strip():
            r = json.loads(l); done_keys.add((r["policy"], r["task"], r["init"], r["condition"]))
with out.open("a") as fh:
    for task_id in range(10):
        task = suite.get_task(task_id); ntask = norm_task(task.language)
        inits_all = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
        env.seed(7)
        for init in inits:
            for cond in conds:
                inst = (suite_name, task_id, init)
                for pol in policies:
                    if (pol, task_id, init, cond) in done_keys:
                        continue
                    env.reset()
                    obs = env.set_init_state(inits_all[init])
                    pert = None
                    if cond.startswith("robot_init:"):
                        pert = ev.perturb_robot_init(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"))
                    if cond.startswith("layout:"):
                        obs, pert = perturb_layout(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"), ev.LIBERO_DUMMY_ACTION,
                                                   reset_fn=lambda: ev._safe_reset(env))
                        if obs is None:
                            fh.write(json.dumps(dict(policy=pol, suite=suite_name, task=task_id, init=init, condition=cond, skipped=True)) + "\n")
                            continue
                    for _ in range(10):
                        obs, _, _, _ = env.step(ev.LIBERO_DUMMY_ACTION)
                    rel = EpisodeRelativeState(); plan = collections.deque(); done = False; t = 0
                    while t < ev.MAX_STEPS[suite_name]:
                        if not plan:
                            s = ev._state(obs)
                            feat = {"T": [t / 300.0], "A": s, "Rl": rel(s)}[pol]
                            if pol != "Rl":
                                rel(s)  # keep the anchor consistent for logging
                            plan.extend(models[pol].act(feat, ntask)[:5])
                        a = np.clip(plan.popleft(), -1, 1)
                        obs, _, done, _ = env.step(a.tolist())
                        t += 1
                        if done:
                            break
                    fh.write(json.dumps(dict(policy=pol, suite=suite_name, task=task_id, init=init, condition=cond,
                                             success=bool(done), steps=t, perturbation=pert and {k: v for k, v in pert.items() if k != "tries"})) + "\n")
                    fh.flush()
        env.close()
        print(suite_name, "task", task_id, round(time.time() - t0), "s", flush=True)
