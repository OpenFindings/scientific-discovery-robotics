"""Ego-centric oracle LIBERO policies (object minus gripper position): task + object-of-interest positions (stand-in for perfect vision)
[+ absolute | episode-relative robot state] -> next 10 actions."""
import glob, re, sys
import numpy as np, torch
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
from relative_state import to_relative
torch.manual_seed(0)
files, out_dir = sorted(glob.glob(sys.argv[1])), sys.argv[2]
H = 10
def norm_task(s): return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")
tasks = [norm_task(str(np.load(f)["task"])) for f in files]
feats = {"E": [], "EA": [], "ERl": []}; Y = []; T = []
for ti, f in enumerate(files):
    d = np.load(f)
    S, O, A, E = d["state"].astype(np.float64), d["obj"].astype(np.float64), d["actions"], d["episode"]
    for e in np.unique(E):
        m = np.where(E == e)[0]
        a = A[m]
        # OpenVLA/openpi no-op filter: drop near-zero arm motion with unchanged gripper command
        prev = np.r_[a[:1, 6], a[:-1, 6]]
        keep = ~((np.abs(a[:, :6]).max(1) < 1e-4) & (a[:, 6] == prev))
        m = m[keep]
        s, o, a = S[m], O[m], A[m]
        n = len(m)
        rel = to_relative(s, s[0]).astype(np.float64)
        eg = o.copy()  # ego-centric: object position minus gripper position (what a wrist camera sees)
        for k in range(2):
            present = np.abs(o[:, 3 * k:3 * k + 3]).sum(1) > 0
            eg[present, 3 * k:3 * k + 3] -= s[present, :3]
        feats["E"].append(eg); feats["EA"].append(np.c_[eg, s]); feats["ERl"].append(np.c_[eg, rel])
        idx = np.minimum(np.arange(n)[:, None] + np.arange(H)[None], n - 1)
        Y.append(a[idx].reshape(n, -1)); T.append(np.full(n, ti))
Y = np.concatenate(Y).astype(np.float32); T = np.concatenate(T)
ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
print("frames", len(Y), "tasks", len(tasks), flush=True)
for name, F in feats.items():
    X = np.concatenate(F).astype(np.float32); xmu, xsd = X.mean(0), X.std(0) + 1e-6
    x = torch.tensor((X - xmu) / xsd); t = torch.tensor(T); y = torch.tensor((Y - ymu) / ysd)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1] + len(tasks), 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                              torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); steps = 15000
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        i = torch.randint(0, len(x), (1024,))
        inp = torch.cat([x[i], torch.nn.functional.one_hot(t[i], len(tasks)).float()], 1)
        loss = ((net(inp) - y[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 5000 == 0 or step == steps - 1: print(name, step, round(loss.item(), 4), flush=True)
    torch.save(dict(state=net.state_dict(), xmu=xmu, xsd=xsd, ymu=ymu, ysd=ysd, tasks=tasks, H=H, in_dim=X.shape[1],
                    n_obj=6), f"{out_dir}/ego_{name}.pt")
print("saved", flush=True)
