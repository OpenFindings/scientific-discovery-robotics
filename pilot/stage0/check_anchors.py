"""Arm C anchor check: do eval episodes start where the training demos started?

Arm C's state is relative to the episode's first state. In training that anchor
is the dataset's frame 0; at evaluation it is the state at the first policy query,
after the 10 settle steps (eval_libero.py logs it as `start_state`, absolute,
for every arm). A systematic difference between the two shifts every relative
state arm C sees at test time, even with no perturbation.

What the training anchors look like (measured on 212 of the 1,693 episodes of
physical-intelligence/libero, every 8th):
  * eef_pos is in the WORLD frame, not the robot base frame. Anchors fall into
    two clusters ~0.9 m apart, one per scene layout. Compare within task only.
  * Within a task, frame-0 positions spread by up to 17-28 mm (median max
    deviation ~10 mm per axis, mostly in a plane); orientation varies < 0.5 deg;
    the gripper is always fully open (0.0388).
So "equal to a training anchor" is the wrong test. The check asks two questions:

  systematic  Is the mean offset of eval starts from their task's mean training
              anchor small? A shared offset is what a settle-vs-frame-0 mismatch
              would produce. Reported in mm and in arm C state-token bins (bin
              width (q99 - q01) / 128 of the relative training states).
  coverage    Does each eval start fall inside its task's training-anchor cloud:
              distance from the task's mean anchor within the pooled 99th
              percentile of training distances from their own task means
              (+2 mm), and orientation within the pooled 99th percentile of
              training orientation deviations (+0.5 deg)? Pooling over the 40
              tasks keeps the reference stable when a task has few demos.
It also checks that the same instance starts identically across arms and seeds,
which the paired design assumes.

Pre-set pass rule:
  |mean offset| <= 5 mm on every axis and <= 1 deg;
  >= 95% of eval instances inside their task's cloud;
  every eval task matched to training demos;
  same-instance starts identical across arms and seeds to 0.1 mm.

Training anchors come from one of:
  --dataset DIR    a local LeRobot v2.0 dataset (e.g. $HF_LEROBOT_HOME/physical-intelligence/libero)
  --hf REPO        the Hugging Face dataset, reading only footers and state columns
                   over HTTP range requests (~2 s per episode; no 30 GB download)
Needs numpy and pyarrow. Runs before or after convert_relative_state.py.

  python check_anchors.py results/stage0/*.jsonl --dataset $HF_LEROBOT_HOME/physical-intelligence/libero \\
      --out results/stage0/anchor_check.json
"""
import argparse
import io
import json
import pathlib
import re
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from relative_state import axis_angle_to_matrix, matrix_to_axis_angle, to_relative  # noqa: E402

THRESH = dict(mean_offset_mm=5.0, mean_offset_deg=1.0, share_inside=0.95, inside_tol_mm=2.0,
              inside_tol_deg=0.5, pairing_mm=0.1)
UNPERTURBED = ("id", "scene_blackout", "wrist_blackout_after_close")


def norm_task(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


# --- training anchors -------------------------------------------------------------------------
class _HTTPRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests (enough for pyarrow's parquet reader)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _episode_states(fobj) -> np.ndarray:
    t = pq.ParquetFile(fobj).read(columns=["state"])
    return np.asarray(t.column("state").to_pylist(), dtype=np.float64)


def load_training(dataset: str = None, hf: str = None, max_episodes: int = None, stride: int = 1):
    if hf:
        base = f"https://huggingface.co/datasets/{hf}/resolve/main"
        fetch = lambda rel: urllib.request.urlopen(f"{base}/{rel}").read().decode()  # noqa: E731
        open_ep = lambda rel: _HTTPRangeFile(f"{base}/{rel}")  # noqa: E731
    else:
        root = pathlib.Path(dataset)
        fetch = lambda rel: (root / rel).read_text()  # noqa: E731
        open_ep = lambda rel: str(root / rel)  # noqa: E731
    info = json.loads(fetch("meta/info.json"))
    episodes = [json.loads(l) for l in fetch("meta/episodes.jsonl").splitlines() if l.strip()]
    episodes = episodes[::stride][:max_episodes] if max_episodes else episodes[::stride]
    out = []
    for i, ep in enumerate(episodes):
        idx = ep["episode_index"]
        rel = info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
        states = _episode_states(open_ep(rel))
        out.append(dict(episode=idx, task=norm_task(ep["tasks"][0]), anchor=states[0],
                        rel=to_relative(states, states[0])))
        if (i + 1) % 100 == 0:
            print(f"  read {i + 1}/{len(episodes)} training episodes", file=sys.stderr, flush=True)
    return out


# --- eval starts --------------------------------------------------------------------------------
def load_eval(paths):
    starts = defaultdict(list)  # (suite, task_id, init) -> [(arm, seed, task, start_state)]
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("perturbation") or r.get("condition") not in UNPERTURBED or not r.get("start_state"):
                continue
            starts[(r["suite"], r["task"], r["init"])].append(
                (r["arm"], r["seed"], norm_task(r["task_language"]), np.asarray(r["start_state"], float)))
    return starts


# --- comparison ---------------------------------------------------------------------------------
def _mean_rotation(aa):
    """Chordal L2 mean of orientations given as axis-angle vectors (n, 3) -> rotation matrix."""
    M = axis_angle_to_matrix(aa).mean(0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def _angle_to(R, aa):
    rel = np.einsum("ji,njk->nik", R, axis_angle_to_matrix(np.atleast_2d(aa)))
    return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def compare(train, starts):
    allrel = np.concatenate([t["rel"] for t in train])
    q01, q99 = np.quantile(allrel, 0.01, 0), np.quantile(allrel, 0.99, 0)
    bin_width = (q99 - q01) / 128.0  # openpi: normalized [-1, 1] cut into 256 bins
    by_task = defaultdict(list)
    for t in train:
        by_task[t["task"]].append(t["anchor"])
    clouds, pooled_mm, pooled_deg = {}, [], []
    for task, A in by_task.items():
        A = np.stack(A)
        mu, Rm = A[:, :3].mean(0), _mean_rotation(A[:, 3:6])
        dmm, ddeg = 1000 * np.linalg.norm(A[:, :3] - mu, axis=1), _angle_to(Rm, A[:, 3:6])
        if len(A) > 1:
            pooled_mm += dmm.tolist()
            pooled_deg += ddeg.tolist()
        clouds[task] = dict(n=len(A), mu=mu, R=Rm, max_mm=float(dmm.max()), max_deg=float(ddeg.max()))
    lim_mm = float(np.percentile(pooled_mm, 99)) + THRESH["inside_tol_mm"] if pooled_mm else THRESH["inside_tol_mm"]
    lim_deg = float(np.percentile(pooled_deg, 99)) + THRESH["inside_tol_deg"] if pooled_deg else THRESH["inside_tol_deg"]

    per_inst, unmatched, pairing = [], set(), []
    for key, recs in sorted(starts.items()):
        ss = np.stack([r[3] for r in recs])
        pairing.append(float(1000 * np.abs(ss[:, :3] - ss[0, :3]).max()))
        task, s = recs[0][2], ss[0]
        c = clouds.get(task)
        if c is None:
            unmatched.add(task)
            continue
        off = s[:3] - c["mu"]
        rot_off = matrix_to_axis_angle(c["R"].T @ axis_angle_to_matrix(s[3:6]))  # in the mean-anchor frame
        dist_mm, deg = float(1000 * np.linalg.norm(off)), float(np.degrees(np.linalg.norm(rot_off)))
        per_inst.append(dict(
            suite=key[0], task_id=key[1], init=key[2], task=task, n_train_demos=c["n"],
            offset_mm=(1000 * off).round(2).tolist(), offset_rotvec_deg=np.degrees(rot_off).round(3).tolist(),
            dist_mm=dist_mm, deg=deg, limit_mm=lim_mm, limit_deg=lim_deg,
            inside=bool(dist_mm <= lim_mm and deg <= lim_deg),
            gripper_open_diff_mm=float(1000 * np.abs(s[6:8] - np.array([0.0388, -0.0388])).max())))
    spread = {t: dict(n=c["n"], max_mm=c["max_mm"], max_deg=c["max_deg"]) for t, c in clouds.items()}
    spread["_limits"] = dict(limit_mm=lim_mm, limit_deg=lim_deg)
    return per_inst, unmatched, pairing, spread, dict(q01=q01.round(5).tolist(), q99=q99.round(5).tolist(),
                                                      bin_width=bin_width.round(6).tolist())


def summarize(per_inst, unmatched, pairing, spread, quant, n_train):
    if per_inst:
        mean_off = np.mean([r["offset_mm"] for r in per_inst], 0)
        mean_rot = np.mean([r["offset_rotvec_deg"] for r in per_inst], 0)
        bw = np.array(quant["bin_width"])
        sys_bins = np.abs(np.concatenate([mean_off / 1000, np.radians(mean_rot)])) / np.maximum(bw[:6], 1e-9)
        share_inside = float(np.mean([r["inside"] for r in per_inst]))
    else:
        mean_off = mean_rot = sys_bins = np.full(3, np.nan)
        share_inside = float("nan")
    checks = dict(
        systematic_pos=bool(per_inst) and bool(np.all(np.abs(mean_off) <= THRESH["mean_offset_mm"])),
        systematic_rot=bool(per_inst) and float(np.linalg.norm(mean_rot)) <= THRESH["mean_offset_deg"],
        coverage=bool(per_inst) and share_inside >= THRESH["share_inside"],
        all_tasks_matched=not unmatched,
        pairing=bool(pairing) and max(pairing) <= THRESH["pairing_mm"])
    limits = spread.pop("_limits")
    sp = list(spread.values())
    return dict(
        pass_=all(checks.values()), checks=checks, thresholds=THRESH,
        n_eval_instances=len(per_inst), n_unmatched_tasks=len(unmatched), n_train_episodes=n_train,
        systematic=dict(mean_offset_mm=np.round(mean_off, 2).tolist(),
                        mean_offset_rotvec_deg=np.round(mean_rot, 3).tolist(),
                        in_C_state_bins=np.round(sys_bins, 2).tolist()),
        coverage=dict(share_inside=round(share_inside, 4),
                      dist_mm_median=float(np.median([r["dist_mm"] for r in per_inst])) if per_inst else None,
                      outside=[r for r in per_inst if not r["inside"]][:10]),
        unmatched_tasks=sorted(unmatched), pairing_max_mm=max(pairing) if pairing else None,
        train_clouds=dict(tasks=len(sp), coverage_limit_mm=round(limits["limit_mm"], 2),
                          coverage_limit_deg=round(limits["limit_deg"], 3),
                          max_mm_median=float(np.median([s["max_mm"] for s in sp])) if sp else None,
                          max_mm_max=float(max(s["max_mm"] for s in sp)) if sp else None,
                          max_deg_max=float(max(s["max_deg"] for s in sp)) if sp else None,
                          demos_per_task_min=min(s["n"] for s in sp) if sp else None),
        relative_state_quantiles=quant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", help="eval_libero.py episode logs (any arms)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="local LeRobot v2.0 dataset directory (absolute state)")
    src.add_argument("--hf", help="Hugging Face dataset repo id, read over HTTP ranges")
    ap.add_argument("--max-train-episodes", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every k-th training episode")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train = load_training(a.dataset, a.hf, a.max_train_episodes, a.stride)
    starts = load_eval(a.jsonl)
    if not starts:
        sys.exit("no unperturbed eval episodes with start_state found")
    rep = summarize(*compare(train, starts), n_train=len(train))
    text = json.dumps(rep, indent=1)
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(text)
    print(text)
    sys.exit(0 if rep["pass_"] else 1)


if __name__ == "__main__":
    main()
