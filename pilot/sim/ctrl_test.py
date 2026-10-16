import libero_boot, pathlib, sys, numpy as np
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
import eval_libero as ev
from layout_perturb import perturb_layout
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
s = benchmark.get_benchmark_dict()["libero_object"](); task = s.get_task(0); inits = s.get_task_init_states(0)
env = ControlEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                 use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
env.seed(7)
def start(mode, init):
    env.env.reset(); obs = env.set_init_state(inits[init])
    if mode != "id":
        obs, rec = perturb_layout(env, 0.03, ev._rng("x", 0, init, "layout:0.03", "dir"), ev.LIBERO_DUMMY_ACTION)
        q_obj = env.env.sim.data.get_joint_qpos(env.env.objects_dict[rec["target"]].joints[-1]).copy()
        if mode == "layout+fresh_reset":
            placed = env.get_sim_state()
            env.env.reset(); obs = env.set_init_state(placed)
    for _ in range(10):
        obs, _, _, _ = env.step(ev.LIBERO_DUMMY_ACTION)
    return ev._state(obs)
for init in range(4):
    a = start("id", init); b = start("layout", init); c = start("layout+fresh_reset", init)
    print(init, "layout vs id mm", round(1000 * np.linalg.norm(b[:3] - a[:3]), 2), "| with fresh reset", round(1000 * np.linalg.norm(c[:3] - a[:3]), 2))
