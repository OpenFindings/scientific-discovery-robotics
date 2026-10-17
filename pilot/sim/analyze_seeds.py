"""Multi-seed paired analysis. Success by arm x condition and paired arm differences:
  pooled  = mean over seeds of the paired difference, 95% CI by task-cluster bootstrap (instances keep all their seeds);
  seeds   = the per-seed point differences, and their SD (seed spread).
Only instances complete for every arm in a given seed are used.

usage: analyze_seeds.py <arms comma> <conds comma> <seed>=<glob> [<seed>=<glob> ...]
Rows may use "policy" or "arm" for the arm key. Arm names may be renamed: "A:R" means arm A reported as R.
"""
import glob, json, sys
from collections import defaultdict
import numpy as np

arm_spec = [a.split(":") for a in sys.argv[1].split(",")]
arms = [a[-1] for a in arm_spec]; rename = {a[0]: a[-1] for a in arm_spec}
conds = sys.argv[2].split(",")
data = defaultdict(dict)  # (seed, suite, task, init, cond) -> arm -> success
skipped = defaultdict(int)
for spec in sys.argv[3:]:
    seed, pat = spec.split("=", 1)
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l); a = rename.get(r.get("arm", r.get("policy")))
            if a is None or r["condition"] not in conds: continue
            if r.get("skipped"): skipped[r["condition"]] += 1; continue
            data[(int(seed), r["suite"], r["task"], r["init"], r["condition"])][a] = float(r["success"])
data = {k: v for k, v in data.items() if all(a in v for a in arms)}
seeds = sorted({k[0] for k in data})
rng = np.random.default_rng(0)
pairs = [(arms[j], arms[i]) for i in range(len(arms)) for j in range(i + 1, len(arms))]


def table(cond):
    ks = [k for k in data if k[4] == cond]
    if not ks: return None
    inst = sorted({k[1:] for k in ks})
    tasks = sorted({(i[0], i[1]) for i in inst})
    out = dict(n_inst=len(inst), n_eps=len(ks), seeds=seeds)
    # per-instance mean over available seeds
    def inst_mean(f):
        m = {}
        for i in inst:
            v = [f(data[(s,) + i]) for s in seeds if (s,) + i in data]
            if v: m[i] = np.mean(v)
        return m
    for a in arms:
        per_seed = [100 * np.mean([data[k][a] for k in ks if k[0] == s]) for s in seeds if any(k[0] == s for k in ks)]
        out[a] = dict(mean=float(np.mean(per_seed)), per_seed=[round(float(x), 1) for x in per_seed])
    for a, b in pairs:
        m = inst_mean(lambda v: v[a] - v[b])
        by_task = defaultdict(list)
        for i, v in m.items(): by_task[(i[0], i[1])].append(v)
        tl = list(by_task.values())
        bs = [np.mean([x for j in rng.integers(0, len(tl), len(tl)) for x in tl[j]]) for _ in range(4000)]
        per_seed = [100 * np.mean([data[k][a] - data[k][b] for k in ks if k[0] == s]) for s in seeds if any(k[0] == s for k in ks)]
        out[f"{a}-{b}"] = dict(pooled=100 * float(np.mean(list(m.values()))), ci=[100 * float(x) for x in np.percentile(bs, [2.5, 97.5])],
                              per_seed=[round(float(x), 1) for x in per_seed], seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None)
    return out


res = {c: table(c) for c in conds}
print(f"seeds {seeds}; skipped (layout, all seeds/arms): {dict(skipped)}")
hdr = f"{'condition':16s} {'inst':>4s} " + " ".join(f"{a:>16s}" for a in arms)
print(hdr)
for c, t in res.items():
    if not t: continue
    print(f"{c:16s} {t['n_inst']:4d} " + " ".join(f"{t[a]['mean']:5.1f} ({'/'.join(f'{x:.0f}' for x in t[a]['per_seed'])})".rjust(16) for a in arms))
print()
for c, t in res.items():
    if not t: continue
    print(c)
    for a, b in pairs:
        d = t[f"{a}-{b}"]
        sd = f"{d['seed_sd']:.1f}" if d["seed_sd"] is not None else "-"
        print(f"   {a}-{b}: {d['pooled']:+6.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]   per seed {d['per_seed']}  seed SD {sd}")
if len(sys.argv) > 3 and "JSON_OUT" in __import__("os").environ:
    json.dump(res, open(__import__("os").environ["JSON_OUT"], "w"), indent=1)
