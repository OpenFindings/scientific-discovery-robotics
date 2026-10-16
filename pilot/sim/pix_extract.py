"""Download agentview + eye_in_hand images of the original LIBERO demos (HDF5 over HTTP, 128x128),
downsample to RES x RES, and store uint8 in the same frame order as oracle_extract.py
(demos sorted by index, all frames)."""
import io, os, sys, time, urllib.request
import numpy as np, h5py, cv2
sys.path.insert(0, os.path.dirname(__file__))
from oracle_extract_reader import R  # HTTP range file
suite, task_id, out, RES = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
import json
bddl = sys.argv[5]  # file name of the task HDF5
base = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"
t0 = time.time()
h = h5py.File(io.BufferedReader(R(f"{base}/{suite}/{bddl}"), 1 << 22), "r")
A, W = [], []
for name in sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1])):
    o = h["data"][name]["obs"]
    for key, dst in (("agentview_rgb", A), ("eye_in_hand_rgb", W)):
        imgs = o[key][()]
        dst.append(np.stack([cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA) for im in imgs]))
h.close()
np.savez(out, agentview=np.concatenate(A), wrist=np.concatenate(W))
print(suite, task_id, sum(len(a) for a in A), "frames", round(time.time() - t0), "s", flush=True)
os._exit(0)
