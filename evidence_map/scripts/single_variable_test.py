"""Does one recorded setting variable explain the sign of the state effect?

Decision rule (fixed before running this script):
  Unit of analysis: 'presence' comparisons (same model with vs without
  proprioceptive input) with a numeric effect. Each record is classed
  hurt (< -3 pp), null (|e| <= 3), or help (> +3 pp). Each paper gets
  total weight 1, so a paper with many rows cannot dominate.
  For a variable, each level predicts its weighted-majority class.
  The variable EXPLAINS the contradiction if
    (a) paper-weighted accuracy >= 0.90 on the non-null records, and
    (b) no level contains counterexamples (non-null records whose sign
        differs from the level's prediction) from 2+ different papers.
  Same test is then run on pairs of variables.
"""
import itertools
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in open(ROOT / "data" / "results.jsonl")]
rows = [r for r in rows if r["comparison"] == "presence" and r["effect_pp"] is not None]


def cls(e):
    return "hurt" if e < -3 else "help" if e > 3 else "null"


def coarse_action(a):
    return "absolute" if a.startswith("absolute") else "relative" if a.startswith("relative") else "delta" if a.startswith("delta") else "mixed"


for r in rows:
    r["cls"] = cls(r["effect_pp"])
    r["action_coarse"] = coarse_action(r["action_space"])
    r["history_gt1"] = r["state_history"] > 1
    r["shift_any"] = "ID" if r["shift"] == "ID" else "shifted"

VARS = ["shift", "shift_any", "state_frame", "fusion", "velocity_terms", "pretrained",
        "platform", "wrist_camera", "action_coarse", "history_gt1"]

npaper = defaultdict(int)
for r in rows:
    npaper[r["paper"]] += 1
for r in rows:
    r["w"] = 1 / npaper[r["paper"]]


def evaluate(keyfn, data):
    groups = defaultdict(list)
    for r in data:
        groups[keyfn(r)].append(r)
    correct = total = 0.0
    detail = {}
    for lvl, rs in groups.items():
        wt = defaultdict(float)
        for r in rs:
            wt[r["cls"]] += r["w"]
        pred = max(("hurt", "help"), key=lambda c: wt[c])
        nonnull = [r for r in rs if r["cls"] != "null"]
        counter = [r for r in nonnull if r["cls"] != pred]
        correct += sum(r["w"] for r in nonnull if r["cls"] == pred)
        total += sum(r["w"] for r in nonnull)
        detail[str(lvl)] = dict(pred=pred, n=len(rs), n_help=sum(r["cls"] == "help" for r in rs),
                                n_hurt=sum(r["cls"] == "hurt" for r in rs),
                                n_null=sum(r["cls"] == "null" for r in rs),
                                counterexample_papers=sorted({r["paper"] for r in counter}))
    acc = correct / total if total else float("nan")
    multi = any(len(d["counterexample_papers"]) >= 2 for d in detail.values())
    return dict(accuracy=round(acc, 3), explains=acc >= 0.90 and not multi, levels=detail)


report = {"n_records": len(rows), "n_papers": len(npaper),
          "class_counts": {c: sum(r["cls"] == c for r in rows) for c in ("hurt", "null", "help")},
          "single": {}, "pairs": {}}
for v in VARS:
    report["single"][v] = evaluate(lambda r, v=v: r[v], rows)
for a, b in itertools.combinations(VARS, 2):
    res = evaluate(lambda r, a=a, b=b: (r[a], r[b]), rows)
    report["pairs"][f"{a} x {b}"] = {k: res[k] for k in ("accuracy", "explains")}

# Naive-fusion subset: the 'raw state input' question without remedies.
naive = [r for r in rows if r["fusion"] == "naive"]
for r in naive:
    r["w"] = 1 / sum(1 for x in naive if x["paper"] == r["paper"])
report["naive_only_single"] = {v: evaluate(lambda r, v=v: r[v], naive) for v in VARS if v != "fusion"}

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "single_variable_test.json").write_text(json.dumps(report, indent=1))
print(f"{report['n_records']} presence records from {report['n_papers']} papers; {report['class_counts']}\n")
print("single variable           acc   explains")
for v, res in sorted(report["single"].items(), key=lambda kv: -kv[1]["accuracy"]):
    print(f"  {v:22s} {res['accuracy']:.3f}  {res['explains']}")
print("\nbest pairs")
for k, res in sorted(report["pairs"].items(), key=lambda kv: -kv[1]["accuracy"])[:6]:
    print(f"  {k:36s} {res['accuracy']:.3f}  {res['explains']}")
print("\nnaive-fusion subset, single variable")
for v, res in sorted(report["naive_only_single"].items(), key=lambda kv: -kv[1]["accuracy"])[:5]:
    print(f"  {v:22s} {res['accuracy']:.3f}  {res['explains']}")
print("\nshift levels:", json.dumps(report["single"]["shift"]["levels"], indent=1))
print("naive shift_any levels:", json.dumps(report["naive_only_single"]["shift_any"]["levels"], indent=1))
