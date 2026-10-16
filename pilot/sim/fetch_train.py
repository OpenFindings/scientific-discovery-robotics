"""Fetch state+actions (no images) of every physical-intelligence/libero episode over HTTP ranges."""
import json, sys, time, urllib.request, numpy as np, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
from check_anchors import _HTTPRangeFile, norm_task
base = "https://huggingface.co/datasets/physical-intelligence/libero/resolve/main"
info = json.loads(urllib.request.urlopen(f"{base}/meta/info.json").read())
eps = [json.loads(l) for l in urllib.request.urlopen(f"{base}/meta/episodes.jsonl").read().decode().splitlines() if l.strip()]
def get(ep):
    idx = ep["episode_index"]
    rel = info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
    for attempt in range(5):
        try:
            t = pq.ParquetFile(_HTTPRangeFile(f"{base}/{rel}")).read(columns=["state", "actions"])
            return idx, norm_task(ep["tasks"][0]), np.asarray(t.column("state").to_pylist(), np.float32), np.asarray(t.column("actions").to_pylist(), np.float32)
        except Exception as e:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"episode {idx} failed")
t0 = time.time(); out = []
with ThreadPoolExecutor(16) as ex:
    for i, r in enumerate(ex.map(get, eps)):
        out.append(r)
        if (i + 1) % 200 == 0: print(i + 1, round(time.time() - t0), flush=True)
idx = np.array([r[0] for r in out]); tasks = [r[1] for r in out]
lens = np.array([len(r[2]) for r in out])
np.savez_compressed(sys.argv[1], episode=idx, lengths=lens, tasks=np.array(tasks),
                    state=np.concatenate([r[2] for r in out]), actions=np.concatenate([r[3] for r in out]))
print("done", len(out), "episodes", int(lens.sum()), "frames", round(time.time() - t0), "s")
