"""Train blind (no-vision) LIBERO policies: task one-hot + {time | absolute state | relative state}
-> next 10 actions. Same action chunking as pi0.5 in the pilot (horizon 10, replan every 5)."""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
from relative_state import to_relative

H = 10
import os
SEED = int(os.environ.get('SEED', '0'))
torch.manual_seed(SEED)
SUF = '' if SEED == 0 else f'_s{SEED}'
d = np.load(sys.argv[1])
out_dir = sys.argv[2]
lens, S, A = d["lengths"], d["state"].astype(np.float64), d["actions"].astype(np.float32)
tasks = [str(t) for t in d["tasks"]]
task_list = sorted(set(tasks)); tid = np.array([task_list.index(t) for t in tasks])
starts = np.r_[0, np.cumsum(lens)[:-1]]
feats = {"T": [], "A": [], "Rl": []}; Y = []; TASK = []
for e, (s0, n) in enumerate(zip(starts, lens)):
    st = S[s0:s0 + n]
    feats["T"].append(np.arange(n)[:, None] / 300.0)
    feats["A"].append(st)
    feats["Rl"].append(to_relative(st, st[0]).astype(np.float64))
    idx = np.minimum(np.arange(n)[:, None] + np.arange(H)[None], n - 1)  # pad chunk with last action
    Y.append(A[s0:s0 + n][idx].reshape(n, -1)); TASK.append(np.full(n, tid[e]))
Y = np.concatenate(Y); TASK = np.concatenate(TASK)
ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
for name in ("T", "A", "Rl"):
    X = np.concatenate(feats[name]).astype(np.float32)
    xmu, xsd = X.mean(0), X.std(0) + 1e-6
    x = torch.tensor((X - xmu) / xsd); t = torch.tensor(TASK); y = torch.tensor((Y - ymu) / ysd, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1] + len(task_list), 512), torch.nn.GELU(),
                              torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(),
                              torch.nn.Linear(512, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    n = len(x); bs = 1024; steps = 15000
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        i = torch.randint(0, n, (bs,))
        inp = torch.cat([x[i], torch.nn.functional.one_hot(t[i], len(task_list)).float()], 1)
        loss = ((net(inp) - y[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0 or step == steps - 1:
            print(name, step, round(loss.item(), 4), flush=True)
    torch.save(dict(state=net.state_dict(), xmu=xmu, xsd=xsd, ymu=ymu, ysd=ysd, tasks=task_list, H=H,
                    in_dim=X.shape[1]), f"{out_dir}/blind_{name}{SUF}.pt")
print("saved")
