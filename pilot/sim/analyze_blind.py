"""Blind-policy results: success by policy x condition, on instances complete for all policies
(paired), with a task-cluster bootstrap CI for policy differences."""
import json, sys, glob
from collections import defaultdict
import numpy as np
rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1])) for l in open(f) if l.strip()]
pols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["T", "A", "Rl"]
pairs = [(pols[i], pols[j]) for i in range(len(pols)) for j in range(i) ][::-1]
by = defaultdict(dict)  # (suite, task, init, cond) -> policy -> success
skipped = defaultdict(int)
for r in rows:
    k = (r["suite"], r["task"], r["init"], r["condition"])
    if r.get("skipped"):
        skipped[r["condition"]] += 1; continue
    by[k][r["policy"]] = r["success"]
complete = {k: v for k, v in by.items() if all(p in v for p in pols)}
conds = ["id", "robot_init:0.1", "robot_init:0.2", "layout:0.03", "layout:0.06", "blackout"]
rng = np.random.default_rng(0)
def boot_diff(keys, a, b, reps=4000):
    tasks = sorted({(k[0], k[1]) for k in keys}); idx = {t: [k for k in keys if (k[0], k[1]) == t] for t in tasks}
    pt = np.mean([complete[k][a] - complete[k][b] for k in keys])
    bs = []
    for _ in range(reps):
        samp = [idx[tasks[i]] for i in rng.integers(0, len(tasks), len(tasks))]
        ks = [k for g in samp for k in g]
        bs.append(np.mean([complete[k][a] - complete[k][b] for k in ks]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{100*pt:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"
print(f"paired instances: {len(complete)} ; skipped layout rows: {dict(skipped)}")
print(f"{'condition':16s} {'n':>4s} " + " ".join(f"{p:>6s}" for p in pols) + "   " + "  ".join(f"{a}-{b:<16s}" for a, b in pairs))
for c in conds:
    ks = [k for k in complete if k[3] == c]
    if not ks: continue
    sr = {p: 100 * np.mean([complete[k][p] for k in ks]) for p in pols}
    print(f"{c:16s} {len(ks):4d} " + " ".join(f"{sr[p]:6.1f}" for p in pols) + "   " +
          "  ".join(boot_diff(ks, a, b) for a, b in pairs))
print("\nby suite (success %, T/A/Rl):")
for s in sorted({k[0] for k in complete}):
    line = []
    for c in conds:
        ks = [k for k in complete if k[0] == s and k[3] == c]
        if ks: line.append(f"{c}: " + "/".join(f"{100*np.mean([complete[k][p] for k in ks]):.0f}" for p in pols) + f" (n={len(ks)})")
    print(f"  {s}: " + "; ".join(line))
