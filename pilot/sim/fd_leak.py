import libero_boot, os, pathlib, sys, mujoco
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
s = benchmark.get_benchmark_dict()["libero_goal"](); task = s.get_task(0)
env = ControlEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                 use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
env.seed(7)
n0 = len(os.listdir("/proc/self/fd"))
for i in range(20):
    env.env.reset()
n1 = len(os.listdir("/proc/self/fd"))
env.close(); del env
import gc; gc.collect()
n2 = len(os.listdir("/proc/self/fd"))
print(f"mujoco {mujoco.__version__}: fds after 1st reset {n0}, after 20 more resets {n1} (+{(n1 - n0) / 20:.1f}/reset), after close+gc {n2}")
