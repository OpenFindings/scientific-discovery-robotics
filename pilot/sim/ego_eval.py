"""Closed-loop eval of oracle-vision policies (O / OA / ORl). 'blackout' = object input replaced by its
training mean from the first step (the oracle analogue of a scene-camera blackout)."""
import libero_boot  # noqa
import collections, json, pathlib, re, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
import eval_libero as ev
from layout_perturb import perturb_layout
from relative_state import EpisodeRelativeState
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
suite_name, model_dir, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
conds = sys.argv[4].split(","); inits = ev._range(sys.argv[5]); policies = sys.argv[6].split(",")
torch.set_num_threads(1)
import os
SEED = int(os.environ.get('SEED', '0'))
SUF = '' if SEED == 0 else f'_s{SEED}'
def norm_task(s): return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")
class Pol:
    def __init__(s, p):
        c = torch.load(p, weights_only=False); s.c = c
        s.net = torch.nn.Sequential(torch.nn.Linear(c["in_dim"] + len(c["tasks"]), 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                                    torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, len(c["ymu"])))
        s.net.load_state_dict(c["state"]); s.net.eval()
    def act(s, feat, task, blackout):
        c = s.c; x = (np.asarray(feat, np.float64) - c["xmu"]) / c["xsd"]
        if blackout: x[:c["n_obj"]] = 0.0  # object input = training mean
        oh = np.zeros(len(c["tasks"])); oh[c["tasks"].index(task)] = 1
        with torch.no_grad(): y = s.net(torch.tensor(np.r_[x, oh], dtype=torch.float32)[None])[0].numpy()
        return (y * c["ysd"] + c["ymu"]).reshape(c["H"], 7)
models = {p: Pol(f"{model_dir}/ego_{p}{SUF}.pt") for p in policies}
suite = benchmark.get_benchmark_dict()[suite_name]()
done_keys = set()
if out.exists():
    for l in out.read_text().splitlines():
        if l.strip(): r = json.loads(l); done_keys.add((r["policy"], r["task"], r["init"], r["condition"]))
t0 = time.time()
with out.open("a") as fh:
    for task_id in range(10):
        task = suite.get_task(task_id); ntask = norm_task(task.language); inits_all = suite.get_task_init_states(task_id)
        env = ControlEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                         use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
        env.seed(7); renv = env.env
        objs = [n for n in renv.obj_of_interest if n in renv.objects_dict][:2]
        joints = [renv.objects_dict[n].joints[-1] for n in objs]
        def objfeat():
            o = np.zeros(6)
            for k, j in enumerate(joints): o[3 * k:3 * k + 3] = renv.sim.data.get_joint_qpos(j)[:3]
            return o
        for init in inits:
            for cond in conds:
                inst = (suite_name, task_id, init)
                for pol in policies:
                    if (pol, task_id, init, cond) in done_keys: continue
                    ev._safe_reset(env); obs = env.set_init_state(inits_all[init]); pert = None
                    if cond.startswith("robot_init:"):
                        pert = ev.perturb_robot_init(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"))
                    if cond.startswith("layout:"):
                        obs, pert = perturb_layout(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"), ev.LIBERO_DUMMY_ACTION,
                                                   reset_fn=lambda: ev._safe_reset(env))
                        if obs is None:
                            fh.write(json.dumps(dict(policy=pol, suite=suite_name, task=task_id, init=init, condition=cond, skipped=True)) + "\n"); continue
                    for _ in range(10): obs, _, _, _ = env.step(ev.LIBERO_DUMMY_ACTION)
                    rel = EpisodeRelativeState(); plan = collections.deque(); done = False; t = 0
                    while t < ev.MAX_STEPS[suite_name]:
                        if not plan:
                            s = ev._state(obs); r_ = rel(s); o = objfeat()
                            eg = o.copy()
                            for k in range(len(joints)): eg[3 * k:3 * k + 3] -= s[:3]
                            feat = {"E": eg, "EA": np.r_[eg, s], "ERl": np.r_[eg, r_]}[pol]
                            plan.extend(models[pol].act(feat, ntask, cond == "blackout")[:5])
                        obs, _, done, _ = env.step(np.clip(plan.popleft(), -1, 1).tolist()); t += 1
                        if done: break
                    fh.write(json.dumps(dict(policy=pol, suite=suite_name, task=task_id, init=init, condition=cond,
                                             success=bool(done), steps=t)) + "\n"); fh.flush()
        env.close()
        print(suite_name, "task", task_id, round(time.time() - t0), "s", flush=True)
