"""Rebuild per-step (robot state, object-of-interest positions, action) for original LIBERO demos by
replaying their saved MuJoCo states in LIBERO. Reads only states/actions from the HF HDF5 files
over HTTP ranges (no images)."""
import libero_boot  # noqa
import io, json, os, pathlib, sys, time, urllib.request
import numpy as np, h5py
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
import eval_libero as ev
from layout_perturb import movable_target
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

class R(io.RawIOBase):
    def __init__(s, url):
        s.url, s.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r: s.size = int(r.headers["Content-Length"])
    def readable(s): return True
    def seekable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0): s.pos = {0: o, 1: s.pos + o, 2: s.size + o}[w]; return s.pos
    def readinto(s, b):
        n = min(len(b), s.size - s.pos)
        if n <= 0: return 0
        for a in range(5):
            try:
                req = urllib.request.Request(s.url, headers={"Range": f"bytes={s.pos}-{s.pos+n-1}"})
                with urllib.request.urlopen(req) as r: d = r.read()
                break
            except Exception:
                time.sleep(2 ** a)
        b[:len(d)] = d; s.pos += len(d); return len(d)

suite_name, out_dir = sys.argv[1], pathlib.Path(sys.argv[2])
suite = benchmark.get_benchmark_dict()[suite_name]()
base = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"
K = 2  # first two objects of interest that are movable objects (xyz each)
t0 = time.time()
for task_id in range(10):
    outf = out_dir / f"{suite_name}_t{task_id}.npz"
    if outf.exists():
        continue
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
    env.seed(7); env.reset()
    renv = env.env
    objs = [n for n in renv.obj_of_interest if n in renv.objects_dict][:K]
    joints = [renv.objects_dict[n].joints[-1] for n in objs]
    h = h5py.File(io.BufferedReader(R(f"{base}/{suite_name}/{task.bddl_file.replace('.bddl', '_demo.hdf5')}"), 1 << 18), "r")
    S, O, A, E = [], [], [], []
    for di, name in enumerate(sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))):
        states = h["data"][name]["states"][()]; actions = h["data"][name]["actions"][()]
        for t in range(len(states)):
            obs = env.set_init_state(states[t])
            S.append(ev._state(obs)); A.append(actions[t]); E.append(di)
            o = np.zeros(3 * K)
            for k, j in enumerate(joints):
                o[3 * k:3 * k + 3] = renv.sim.data.get_joint_qpos(j)[:3]
            O.append(o)
    h.close(); env.close()
    np.savez_compressed(outf, state=np.array(S, np.float32), obj=np.array(O, np.float32), actions=np.array(A, np.float32),
                        episode=np.array(E), objects=np.array(objs + [""] * (K - len(objs))), task=task.language)
    print(suite_name, task_id, len(S), "frames", objs, round(time.time() - t0), "s", flush=True)
print("done", flush=True)
os._exit(0)
