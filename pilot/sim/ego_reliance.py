"""Same probe for the ego-oracle MLPs: RMS change of the normalized action chunk when the object input or the
state input is replaced by its training mean (normalized 0), on 2,000 random training frames (no-op filtered)."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
from relative_state import to_relative
from pix_train import norm_task
d0 = sys.argv[1]; model_dir = sys.argv[2]
meta = np.load(f"{d0}/meta.npz")
files = sorted(glob.glob(sys.argv[3]))
# rebuild ego features on the packed (filtered) frames: obj columns come from the oracle files in the same order
O = []
for f in files:
    d = np.load(f); S, Ob, A, E = d["state"].astype(np.float64), d["obj"].astype(np.float64), d["actions"], d["episode"]
    for e in np.unique(E):
        m = np.where(E == e)[0]; a = A[m]; prev = np.r_[a[:1, 6], a[:-1, 6]]
        m = m[~((np.abs(a[:, :6]).max(1) < 1e-4) & (a[:, 6] == prev))]
        o = Ob[m].copy(); s = S[m]
        for k in range(2):
            pr = np.abs(o[:, 3 * k:3 * k + 3]).sum(1) > 0; o[pr, 3 * k:3 * k + 3] -= s[pr, :3]
        O.append(o)
O = np.concatenate(O); assert len(O) == len(meta["S"])
i = np.sort(np.random.default_rng(0).choice(len(O), 2000, replace=False))
feats = {"E": O[i], "EA": np.c_[O[i], meta["S"][i]], "ERl": np.c_[O[i], meta["Rl"][i]]}
out = {}
for p in sorted(glob.glob(f"{model_dir}/ego_*.pt")):
    c = torch.load(p, weights_only=False); name = os.path.basename(p)[4:-3].split("_s")[0]
    net = torch.nn.Sequential(torch.nn.Linear(c["in_dim"] + len(c["tasks"]), 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                              torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, len(c["ymu"])))
    net.load_state_dict(c["state"]); net.eval()
    x = torch.tensor((feats[name] - c["xmu"]) / c["xsd"], dtype=torch.float32)
    oh = torch.nn.functional.one_hot(torch.tensor(meta["T"][i]), len(c["tasks"])).float()
    run = lambda z: net(torch.cat([z, oh], 1))
    with torch.no_grad():
        b = run(x); r = {}
        z = x.clone(); z[:, :6] = 0; r["object"] = float(((run(z) - b) ** 2).mean().sqrt())
        if x.shape[1] > 6: z = x.clone(); z[:, 6:] = 0; r["state"] = float(((run(z) - b) ** 2).mean().sqrt())
    out[os.path.basename(p)] = {k: round(v, 3) for k, v in r.items()}; print(os.path.basename(p), out[os.path.basename(p)])
json.dump(out, open(f"{model_dir}/reliance.json", "w"), indent=1)
