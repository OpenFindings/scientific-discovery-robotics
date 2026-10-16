"""Stage 0 analysis: validity checks, replication anchor, seed spread, state saturation.

Reads the per-episode JSONL files written by eval_libero.py and applies
the Stage 0 exit criteria from research/pilot_3x2_liberoplus.md (section 6, and section 8):

  check 1  every trained policy >= 85% in distribution; arm gap reported
  check 2  anchor: R - N under scene blackout on Spatial/Object/Goal >= +10 pp
           (Blackout vs Freeze: +19.8 pp, one seed per arm)
  check 3  configuration shift bites: R - N under robot_init or layout < -3 pp
  check 5  under each shifted condition, N and R are not both < 10% or both > 90%
  prompt   arm N prompts contain no "State:", arm R prompts all do
  seeds    between-seed SD per condition -> seed count for Stage 1
  check 4  (when arm C is present) >= 95% of C's state vectors under robot_init lie inside
           C's training 1st-99th percentile box
  tokens   share of state tokens saturated (bin 255) or out of range (-1), per arm with
           that arm's norm stats (arms without their own stats are scored with R's)

Pure numpy. Usage:
  python analyze_stage0.py results/stage0/*.jsonl \
      --norm-stats R=assets/pilot_R/physical-intelligence/libero/norm_stats.json \
      --norm-stats C=assets/pilot_C/local/libero_relstate/norm_stats.json \
      --out results/stage0/summary.json
"""
import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

TRAINED = ("N", "R")
SOG = ("libero_spatial", "libero_object", "libero_goal")
PRIMARY = ("id", "scene_blackout", "robot_init:0.2")
EXPECT_STATE_IN_PROMPT = {"N": False, "R": True, "C": True, "released": False, "base_probe": True}


def load(paths):
    """Episode rows; layout instances that were skipped (no movable target / no valid placement)
    are not episodes and are dropped here (counted in the report)."""
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
    skipped = defaultdict(int)
    for r in rows:
        if r.get("skipped"):
            skipped[(r["arm"], r["condition"])] += 1
    return [r for r in rows if not r.get("skipped")], skipped


def rate(rs):
    return float(np.mean([r["success"] for r in rs])) if rs else float("nan")


def seed_means(rows, arm, cond, suites=None):
    by = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r["condition"] == cond and (suites is None or r["suite"] in suites):
            by[r["seed"]].append(r["success"])
    return {s: float(np.mean(v)) for s, v in sorted(by.items())}


def diff_ci(rows, a, b, cond, suites=None, reps=2000, rng=None):
    """R-N style difference with a two-level cluster bootstrap: seeds within arm, tasks within suite."""
    rng = rng or np.random.default_rng(0)

    def table(arm):
        t = defaultdict(lambda: defaultdict(list))  # seed -> (suite, task) -> outcomes
        for r in rows:
            if r["arm"] == arm and r["condition"] == cond and (suites is None or r["suite"] in suites):
                t[r["seed"]][(r["suite"], r["task"])].append(r["success"])
        return t

    ta, tb = table(a), table(b)
    if not ta or not tb:
        return None
    tasks = sorted({k for t in (ta, tb) for s in t.values() for k in s})
    by_suite = defaultdict(list)
    for k in tasks:
        by_suite[k[0]].append(k)

    def mean(t, seeds, task_sample):
        vals = [np.mean(t[s][k]) for s in seeds for k in task_sample if k in t[s]]
        return float(np.mean(vals)) if vals else float("nan")

    point = mean(ta, list(ta), tasks) - mean(tb, list(tb), tasks)
    boots = []
    for _ in range(reps):
        sample = []
        for ks in by_suite.values():
            sample += [ks[i] for i in rng.integers(0, len(ks), len(ks))]
        sa = rng.choice(list(ta), len(ta)).tolist()
        sb = rng.choice(list(tb), len(tb)).tolist()
        boots.append(mean(ta, sa, sample) - mean(tb, sb, sample))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(diff_pp=round(100 * point, 1), ci95_pp=[round(100 * lo, 1), round(100 * hi, 1)],
                n_seeds=[len(ta), len(tb)], n_tasks=len(tasks))


def _quantiles(path):
    ns = json.loads(pathlib.Path(path).read_text())["norm_stats"]["state"]
    return np.array(ns["q01"])[:8], np.array(ns["q99"])[:8]


def token_stats(rows, stats_by_arm):
    """stats_by_arm: {arm: norm_stats.json path}; arms without an entry use R's."""
    out = {}
    groups = defaultdict(list)
    for r in rows:
        if r.get("states"):
            groups[(r["arm"], r["condition"])].append(np.array(r["states"]))
    for (arm, cond), arrs in sorted(groups.items()):
        path = stats_by_arm.get(arm) or stats_by_arm.get("R")
        if path is None:
            continue
        q01, q99 = _quantiles(path)
        x = np.concatenate(arrs)
        norm = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0  # openpi _normalize_quantile, no clipping
        tok = np.digitize(norm, bins=np.linspace(-1, 1, 257)[:-1]) - 1  # openpi PaligemmaTokenizer
        first = np.stack([a[0] for a in arrs])
        fnorm = (first - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        out[f"{arm}|{cond}"] = dict(
            share_tokens_saturated_255=round(float((tok == 255).mean()), 4),
            share_tokens_below_range=round(float((tok == -1).mean()), 4),
            share_first_states_outside_q01_q99=round(float((np.abs(fnorm) > 1).any(axis=1).mean()), 4),
            share_states_inside_q01_q99=round(float((np.abs(norm) <= 1).all(axis=1).mean()), 4),
            stats_from="own" if arm in stats_by_arm else "R")
    return out


def seed_sd(rows):
    sds = {}
    for cond in sorted({r["condition"] for r in rows}):
        per_arm = []
        for arm in TRAINED:
            m = list(seed_means(rows, arm, cond).values())
            if len(m) >= 2:
                per_arm.append(np.var(m, ddof=1))
        if per_arm:
            sds[cond] = round(100 * float(np.sqrt(np.mean(per_arm))), 2)
    return sds


def recommend_seeds(sds):
    fault_sd = max([v for k, v in sds.items() if k != "id"] or [float("nan")])
    if np.isnan(fault_sd):
        return dict(seeds=None, reason="need >= 2 seeds per arm")
    if fault_sd <= 4:
        return dict(seeds=4, reason=f"max shifted-condition seed SD {fault_sd} pp <= 4")
    if fault_sd <= 6:
        return dict(seeds=5, reason=f"max shifted-condition seed SD {fault_sd} pp in (4, 6]")
    return dict(seeds="6-7", reason=f"max shifted-condition seed SD {fault_sd} pp > 6; "
                                     "or accept detecting only >= 20 pp effects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--norm-stats", action="append", default=[],
                    help="ARM=norm_stats.json (repeatable); a bare path means R")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, skipped = load(a.jsonl)
    arms = sorted({r["arm"] for r in rows})
    conds = sorted({r["condition"] for r in rows})
    rep = dict(n_episodes=len(rows), arms=arms, conditions=conds,
               errors=sum(1 for r in rows if r.get("error")),
               skipped_layout_instances={f"{a_}|{c}": n for (a_, c), n in sorted(skipped.items())})

    rep["success"] = {arm: {c: {s: round(100 * v, 1) for s, v in seed_means(rows, arm, c).items()} for c in conds}
                      for arm in arms}
    rep["success_by_suite"] = {f"{arm}|{c}|{su}": round(100 * rate(
        [r for r in rows if r["arm"] == arm and r["condition"] == c and r["suite"] == su]), 1)
        for arm in arms for c in conds for su in sorted({r["suite"] for r in rows})
        if any(r["arm"] == arm and r["condition"] == c and r["suite"] == su for r in rows)}

    prompt = {}
    for arm in arms:
        ps = [r.get("prompt") or "" for r in rows if r["arm"] == arm]
        has = float(np.mean(["State:" in p for p in ps])) if ps else float("nan")
        exp = EXPECT_STATE_IN_PROMPT.get(arm)
        prompt[arm] = dict(share_with_state=round(has, 3), expected=exp,
                           ok=None if exp is None else (has == 1.0 if exp else has == 0.0),
                           example=ps[0][:200] if ps else None)
    rep["prompt_check"] = prompt

    checks = {}
    if all(arm in arms for arm in TRAINED):
        id_ok = {f"{arm}|{s}": v for arm in TRAINED for s, v in rep["success"][arm].get("id", {}).items()}
        gap = diff_ci(rows, "R", "N", "id")
        checks["1_converged"] = dict(pass_=all(v >= 85 for v in id_ok.values()), id_success=id_ok,
                                     R_minus_N_id=gap, note="gap > 5 pp is reported, not blocking")
        anchor = diff_ci(rows, "R", "N", "scene_blackout", suites=SOG)
        checks["2_anchor"] = dict(pass_=bool(anchor and anchor["diff_pp"] >= 10), R_minus_N=anchor,
                                  reference="Blackout vs Freeze Table 5: +19.8 pp (n=450, 1 seed/arm)")
        robot = [c for c in conds if c.startswith(("robot_init", "layout"))]
        bites = {c: diff_ci(rows, "R", "N", c) for c in robot}
        checks["3_config_shift_bites"] = dict(pass_=any(v and v["diff_pp"] < -3 for v in bites.values()),
                                              R_minus_N=bites)
        # A contrast is unmeasurable only if BOTH arms sit at the floor (< 10%) or the ceiling (> 90%).
        # N alone is expected near 9% under scene blackout (Blackout vs Freeze), so N is not tested alone.
        pairs = {c: [round(100 * rate([r for r in rows if r["arm"] == arm and r["condition"] == c]), 1)
                     for arm in TRAINED] for c in conds if c != "id"}
        checks["5_no_floor_ceiling"] = dict(
            pass_=all(not (max(v) < 10 or min(v) > 90) for v in pairs.values()),
            N_R_success=pairs)
        checks["prompt_flags"] = dict(pass_=all(prompt[a_]["ok"] for a_ in TRAINED if a_ in prompt))
    rep["checks"] = checks
    rep["stage0_exit"] = bool(checks) and all(v["pass_"] for v in checks.values())

    rep["seed_sd_pp"] = seed_sd(rows)
    rep["seed_recommendation"] = recommend_seeds(rep["seed_sd_pp"])
    pert = [r["perturbation"] for r in rows if r.get("perturbation")]
    if pert:
        rep["robot_init_applied"] = dict(n=len(pert), mean_norm=round(float(np.mean([p["applied_norm"] for p in pert])), 4),
                                         share_clipped=round(float(np.mean([p["clipped"] for p in pert])), 3))
    stats_by_arm = dict(x.split("=", 1) if "=" in x else ("R", x) for x in a.norm_stats)
    if stats_by_arm:
        rep["state_tokens"] = token_stats(rows, stats_by_arm)
        if "C" in arms and "C" in stats_by_arm:
            inside = {k.split("|")[1]: v["share_states_inside_q01_q99"] for k, v in rep["state_tokens"].items()
                      if k.startswith("C|robot_init")}
            ref = {k.split("|")[1]: v["share_states_inside_q01_q99"] for k, v in rep["state_tokens"].items()
                   if k.startswith("R|robot_init")}
            checks["4_C_in_distribution"] = dict(pass_=bool(inside) and all(v >= 0.95 for v in inside.values()),
                                                 C_share_inside=inside, R_share_inside_for_contrast=ref)
            rep["stage0_exit"] = all(v["pass_"] for v in checks.values())

    text = json.dumps(rep, indent=1)
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
