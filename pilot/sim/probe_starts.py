"""Policy-free LIBERO probe: start states under ID, robot_init and layout, with the pilot's own
perturbation code, no rendering. Writes eval_libero-format JSONL rows (arm='sim')."""
import libero_boot  # noqa: F401  (must precede LIBERO imports)
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
import eval_libero as ev
from layout_perturb import perturb_layout, movable_target
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

suite_name, out = sys.argv[1], pathlib.Path(sys.argv[2])
conds = sys.argv[3].split(",") if len(sys.argv) > 3 else ["id", "robot_init:0.1", "robot_init:0.2", "robot_init:0.3", "layout:0.03", "layout:0.06"]
suite = benchmark.get_benchmark_dict()[suite_name]()
t0 = time.time()
done_keys = set()
if out.exists():
    for l in out.read_text().splitlines():
        if l.strip():
            r = json.loads(l); done_keys.add((r["task"], r["init"], r["condition"]))
with out.open("a") as fh:
    for task_id in range(10):
        task = suite.get_task(task_id)
        inits = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
        env.seed(7)
        for init in range(10):
            for cond in conds:
                inst = (suite_name, task_id, init)
                if (task_id, init, cond) in done_keys:
                    continue
                env.reset()
                obs = env.set_init_state(inits[init])
                rec = dict(perturbation=None)
                if cond.startswith("robot_init:"):
                    rec["perturbation"] = ev.perturb_robot_init(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"))
                if cond.startswith("layout:"):
                    obs, rec["perturbation"] = perturb_layout(env, float(cond.split(":")[1]), ev._rng(*inst, cond, "dir"), ev.LIBERO_DUMMY_ACTION)
                    if obs is None:
                        fh.write(json.dumps(dict(arm="sim", seed="0", suite=suite_name, task=task_id, task_language=task.language,
                                                 init=init, condition=cond, skipped=True, **rec)) + "\n")
                        continue
                for _ in range(10):
                    obs, _, done, _ = env.step(ev.LIBERO_DUMMY_ACTION)
                tgt = movable_target(env.env)
                tpos = env.env.sim.data.get_joint_qpos(env.env.objects_dict[tgt].joints[-1])[:3].tolist() if tgt else None
                fh.write(json.dumps(dict(arm="sim", seed="0", suite=suite_name, task=task_id, task_language=task.language,
                                         init=init, condition=cond, start_state=ev._state(obs).round(6).tolist(),
                                         target=tgt, target_pos_after_settle=tpos, solved_at_start=bool(env.check_success()),
                                         **rec)) + "\n")
                fh.flush()
        env.close()
        print(suite_name, "task", task_id, "done", round(time.time() - t0), "s", flush=True)
