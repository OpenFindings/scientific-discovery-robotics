"""Render-alignment check: replay demo_0's first stored MuJoCo state in our env and compare our render
with the dataset's agentview / eye_in_hand image (128x128)."""
import libero_boot  # noqa
import io, pathlib, sys
import numpy as np, h5py
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
from oracle_extract_reader import R
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
suite_name, task_id, fname, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
base = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"
h = h5py.File(io.BufferedReader(R(f"{base}/{suite_name}/{fname}"), 1 << 20), "r")
d = h["data"]["demo_0"]; st = d["states"][:40]; av = d["obs"]["agentview_rgb"][:40]; wr = d["obs"]["eye_in_hand_rgb"][:40]
print("env_args", h["data"].attrs.get("env_args", "")[:400])
suite = benchmark.get_benchmark_dict()[suite_name](); task = suite.get_task(task_id)
env = ControlEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                 use_camera_obs=False, has_offscreen_renderer=True, has_renderer=False)
env.seed(0); env.reset()
res = {}
for k in (0, 20, 39):
    env.set_init_state(st[k]); env.env.sim.forward()
    a = env.env.sim.render(camera_name="agentview", width=128, height=128)
    w = env.env.sim.render(camera_name="robot0_eye_in_hand", width=128, height=128)
    for nm, ours, ref in (("av", a, av[k]), ("wr", w, wr[k])):
        for flip in ("none", "ud", "udlr"):
            o = {"none": ours, "ud": ours[::-1], "udlr": ours[::-1, ::-1]}[flip]
            res[(k, nm, flip)] = float(np.abs(o.astype(float) - ref.astype(float)).mean())
    np.savez(f"{out}_{k}.npz", ours_av=a, ref_av=av[k], ours_wr=w, ref_wr=wr[k])
for k, v in sorted(res.items()): print(k, round(v, 2))
import os; os._exit(0)
