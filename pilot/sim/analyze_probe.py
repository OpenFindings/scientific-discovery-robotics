"""Summarize the policy-free LIBERO probe (system python: numpy 2 + pyarrow)."""
import json, sys, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/user/scientific-discovery-robotics/pilot/stage0")
from relative_state import to_relative

probe_dir, train_npz = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for f in sorted(glob.glob(f"{probe_dir}/*.jsonl")) for l in open(f) if l.strip()]
d = np.load(train_npz)
lens, S = d["lengths"], d["state"].astype(np.float64)
starts = np.r_[0, np.cumsum(lens)[:-1]]
rel_all = np.concatenate([to_relative(S[s:s + n], S[s]) for s, n in zip(starts, lens)])
q = {"R": (np.quantile(S, .01, 0), np.quantile(S, .99, 0)), "C": (np.quantile(rel_all, .01, 0), np.quantile(rel_all, .99, 0))}
rep = {"n_rows": len(rows)}

def tokens(x, arm):
    lo, hi = q[arm]
    n = (x - lo) / (hi - lo + 1e-6) * 2 - 1
    return np.digitize(n, np.linspace(-1, 1, 257)[:-1]) - 1

# start-state token range: R uses absolute start state; C's start state is exactly 0 by construction,
# so for C we score the state after a typical first motion is irrelevant -> report C at t=0 (zeros).
by_cond = defaultdict(list)
for r in rows:
    if r.get("start_state"):
        by_cond[r["condition"]].append(np.array(r["start_state"]))
out = {}
idm = {(r["suite"], r["task"], r["init"]): np.array(r["start_state"]) for r in rows if r["condition"] == "id" and r.get("start_state")}
for c, xs in sorted(by_cond.items()):
    X = np.stack(xs)
    tR = tokens(X, "R")
    tC = tokens(np.zeros_like(X), "C")
    ent = dict(n=len(X),
               R_share_start_vectors_outside_q01_q99=round(float(((tR < 0) | (tR > 255) | (X < q["R"][0]) | (X > q["R"][1])).any(1).mean()), 3),
               R_share_dims_saturated_or_below=round(float(((tR == 255) | (tR == -1)).mean()), 3),
               C_start_inside=bool(((tC >= 0) & (tC <= 255)).all()))
    if c != "id":
        disp = [1000 * np.linalg.norm(np.array(r["start_state"])[:3] - idm[(r["suite"], r["task"], r["init"])][:3])
                for r in rows if r["condition"] == c and r.get("start_state") and (r["suite"], r["task"], r["init"]) in idm]
        ent["eef_start_displacement_mm_vs_id"] = dict(median=round(float(np.median(disp)), 1), p90=round(float(np.percentile(disp, 90)), 1),
                                                      max=round(float(np.max(disp)), 1))
    out[c] = ent
rep["start_states"] = out
pr = [r["perturbation"] for r in rows if r["condition"].startswith("robot_init") and r.get("perturbation")]
rep["robot_init_clipped_share"] = round(float(np.mean([p["clipped"] for p in pr])), 3) if pr else None
lay = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["condition"].startswith("layout"):
        p = r["perturbation"]
        lay[r["condition"]][r["suite"]].append(r)
rep["layout"] = {}
for c, su in sorted(lay.items()):
    ent = {}
    for s, rs in sorted(su.items()):
        sk = [r for r in rs if r.get("skipped")]
        na = [r for r in sk if not r["perturbation"].get("applicable")]
        ok = [r for r in rs if not r.get("skipped")]
        ent[s] = dict(n=len(rs), valid=len(ok), skipped_no_target=len(na), skipped_no_valid_placement=len(sk) - len(na),
                      mean_tries_valid=round(float(np.mean([r["perturbation"]["n_tries"] for r in ok])), 2) if ok else None,
                      solved_at_start=sum(r.get("solved_at_start", False) for r in ok),
                      targets=sorted({r["perturbation"].get("target") for r in rs if r["perturbation"].get("target")}))
    rep["layout"][c] = ent
rep["solved_at_start_any"] = {c: sum(r.get("solved_at_start", False) for r in rows if r["condition"] == c) for c in sorted({r["condition"] for r in rows})}
print(json.dumps(rep, indent=1))
