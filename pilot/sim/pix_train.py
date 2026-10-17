"""Pixel LIBERO policies: agentview + wrist images (64x64, from the original LIBERO HDF5 demos) + task one-hot
[+ absolute | episode-relative robot state] -> next 10 actions. Replaces the object-coordinate oracle with real vision.

usage: SEED=s pix_train.py <oracle_dir> <pix_dir> <out_dir> <arm N|R|C> [steps]
Frames are aligned by construction: oracle_extract.py and pix_extract.py both walk demos sorted by index, all frames.
A packed uint8 image cache (memmap, shared by concurrent runs through the page cache) is built once in <pix_dir>/pack.
"""
import glob, os, re, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
from relative_state import to_relative

SEED = int(os.environ.get("SEED", "0"))
H, BATCH, RES = 10, 64, 64


def norm_task(s): return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def pack(oracle_dir, pix_dir):
    """No-op filter (as oracle/ego training), chunk targets, absolute + relative state, image cache."""
    pk = f"{pix_dir}/pack"; os.makedirs(pk, exist_ok=True)
    if os.path.exists(f"{pk}/meta.npz"):
        return pk
    files = sorted(glob.glob(f"{oracle_dir}/libero_*_t*.npz"))
    tasks, keep_all, S, Rl, Y, T = [], [], [], [], [], []
    n_img = 0
    for ti, f in enumerate(files):
        d = np.load(f); px = np.load(f"{pix_dir}/{os.path.basename(f)}")
        assert len(px["agentview"]) == len(d["state"]), (f, len(px["agentview"]), len(d["state"]))
        tasks.append(norm_task(str(d["task"])))
        St, A, E = d["state"].astype(np.float64), d["actions"], d["episode"]
        for e in np.unique(E):
            m = np.where(E == e)[0]; a = A[m]
            prev = np.r_[a[:1, 6], a[:-1, 6]]
            m = m[~((np.abs(a[:, :6]).max(1) < 1e-4) & (a[:, 6] == prev))]
            s, a = St[m], A[m]; n = len(m)
            idx = np.minimum(np.arange(n)[:, None] + np.arange(H)[None], n - 1)
            S.append(s); Rl.append(to_relative(s, s[0])); Y.append(a[idx].reshape(n, -1)); T.append(np.full(n, ti))
            keep_all.append(n_img + m)
        n_img += len(d["state"])
    keep = np.concatenate(keep_all)
    img = np.lib.format.open_memmap(f"{pk}/img.npy", "w+", np.uint8, (len(keep), 2, RES, RES, 3))
    off, w = 0, 0
    for f in files:
        px = np.load(f"{pix_dir}/{os.path.basename(f)}"); n = len(px["agentview"])
        k = keep[(keep >= off) & (keep < off + n)] - off
        img[w:w + len(k), 0] = px["agentview"][k]; img[w:w + len(k), 1] = px["wrist"][k]
        w += len(k); off += n
    img.flush(); assert w == len(keep)
    np.savez(f"{pk}/meta.npz", S=np.concatenate(S), Rl=np.concatenate(Rl), Y=np.concatenate(Y).astype(np.float32),
             T=np.concatenate(T), tasks=np.array(tasks))
    return pk


class Enc(nn.Module):
    def __init__(s, out=128):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.ReLU(),
                            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.ReLU())
        s.fc = nn.Linear(64 * 8 * 8, out)

    def forward(s, x):  # x float in [0,1], (B,3,64,64)
        return F.relu(s.fc(s.c(x - 0.5).flatten(1)))


class PixPolicy(nn.Module):
    def __init__(s, n_tasks, state_dim, out_dim):
        super().__init__()
        s.av, s.wr = Enc(), Enc()
        s.head = nn.Sequential(nn.Linear(256 + n_tasks + state_dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                               nn.Linear(512, out_dim))

    def forward(s, av, wr, task_oh, state):
        return s.head(torch.cat([s.av(av), s.wr(wr), task_oh, state], 1))


def to_float(img_u8):  # (B,H,W,3) uint8 -> (B,3,H,W) float
    return torch.from_numpy(np.ascontiguousarray(img_u8)).permute(0, 3, 1, 2).float() / 255.0


def random_shift(x, pad=4):  # DrQ-style augmentation
    B = x.shape[0]; x = F.pad(x, (pad,) * 4, mode="replicate")
    i, j = np.random.randint(0, 2 * pad + 1, B), np.random.randint(0, 2 * pad + 1, B)
    return torch.stack([x[b, :, i[b]:i[b] + RES, j[b]:j[b] + RES] for b in range(B)])


if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    oracle_dir, pix_dir, out_dir, arm = sys.argv[1:5]
    STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
    pk = pack(oracle_dir, pix_dir); meta = np.load(f"{pk}/meta.npz"); img = np.load(f"{pk}/img.npy", mmap_mode="r")
    tasks = [str(t) for t in meta["tasks"]]; Y, T = meta["Y"], meta["T"]
    X = {"N": np.zeros((len(Y), 0)), "R": meta["S"], "C": meta["Rl"]}[arm].astype(np.float32)
    xmu, xsd = X.mean(0), X.std(0) + 1e-6; ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    x = torch.tensor((X - xmu) / xsd); y = torch.tensor((Y - ymu) / ysd); t = torch.tensor(T)
    net = PixPolicy(len(tasks), X.shape[1], Y.shape[1])
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    print(arm, "seed", SEED, "frames", len(Y), "tasks", len(tasks), "params", sum(p.numel() for p in net.parameters()), flush=True)
    suf = "" if SEED == 0 else f"_s{SEED}"
    ck = f"{out_dir}/ckpt_{arm}{suf}.pt"; start = 0
    if os.path.exists(ck):  # resume after a container restart
        c = torch.load(ck, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sch.load_state_dict(c["sch"])
        torch.set_rng_state(c["torch_rng"]); np.random.set_state(c["np_rng"]); start = c["step"]
        print("resumed at", start, flush=True)
    t0 = time.time(); run = 0.0
    for step in range(start, STEPS):
        i = np.sort(np.random.randint(0, len(Y), BATCH))  # sorted: friendlier memmap reads
        im = img[i]
        av, wr = random_shift(to_float(im[:, 0])), random_shift(to_float(im[:, 1]))
        ii = torch.from_numpy(i)
        loss = ((net(av, wr, F.one_hot(t[ii], len(tasks)).float(), x[ii]) - y[ii]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 2000 == 1999:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), torch_rng=torch.get_rng_state(),
                            np_rng=np.random.get_state(), step=step + 1), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        if step % 500 == 0 or step == STEPS - 1:
            print(arm, step, round(run, 4), round(time.time() - t0), "s", flush=True)
    torch.save(dict(state=net.state_dict(), xmu=xmu, xsd=xsd, ymu=ymu, ysd=ysd, tasks=tasks, H=H, state_dim=X.shape[1],
                    arm=arm, steps=STEPS), f"{out_dir}/pix_{arm}{suf}.pt")
    if os.path.exists(ck): os.remove(ck)
    print("saved", flush=True)
