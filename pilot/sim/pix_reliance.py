"""Offline input-reliance probe for pixel policies: on 2,000 random training frames, the RMS change of the
normalized action chunk when one input is replaced: state -> training mean, agentview -> zeros, wrist -> zeros.
Reported relative to the RMS of the normalized targets (1.0 = as large as the action variation itself)."""
import glob, os, sys, json
import numpy as np, torch, torch.nn.functional as F
from pix_train import PixPolicy, to_float
torch.set_num_threads(1)
pk, model_dir = sys.argv[1], sys.argv[2]
meta = np.load(f"{pk}/meta.npz"); img = np.load(f"{pk}/img.npy", mmap_mode="r")
rng = np.random.default_rng(0); i = np.sort(rng.choice(len(meta["Y"]), 2000, replace=False))
im = img[i]; av, wr = to_float(im[:, 0]), to_float(im[:, 1]); T = torch.tensor(meta["T"][i])
out = {}
for p in sorted(glob.glob(f"{model_dir}/pix_*.pt")):
    c = torch.load(p, weights_only=False); arm = c["arm"]
    net = PixPolicy(len(c["tasks"]), c["state_dim"], len(c["ymu"])); net.load_state_dict(c["state"]); net.eval()
    X = {"N": np.zeros((len(i), 0)), "R": meta["S"][i], "C": meta["Rl"][i], "D": meta["S"][i]}[arm]
    x = torch.tensor((X - c["xmu"]) / c["xsd"], dtype=torch.float32); oh = F.one_hot(T, len(c["tasks"])).float()
    y = torch.tensor((meta["Y"][i] - c["ymu"]) / c["ysd"], dtype=torch.float32)
    with torch.no_grad():
        base = net(av, wr, oh, x); r = {"fit_mse": float(((base - y) ** 2).mean())}
        if c["state_dim"]: r["state"] = float(((net(av, wr, oh, torch.zeros_like(x)) - base) ** 2).mean().sqrt())
        r["agentview"] = float(((net(torch.zeros_like(av), wr, oh, x) - base) ** 2).mean().sqrt())
        r["wrist"] = float(((net(av, torch.zeros_like(wr), oh, x) - base) ** 2).mean().sqrt())
    out[os.path.basename(p)] = {k: round(v, 3) for k, v in r.items()}; print(os.path.basename(p), out[os.path.basename(p)], flush=True)
json.dump(out, open(f"{model_dir}/reliance.json", "w"), indent=1)
