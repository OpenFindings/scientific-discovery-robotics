"""Load the proprioception evidence map into the Paper 4 planner and plan.

Converts data/results.jsonl into the planner's inputs by transparent
rules (below), then runs gap analysis and action ranking with the
Paper 3-informed utility weights. Requires the sibling checkout
../scientific-evidence-planner (Paper 4) for the `planner` package.

Graph construction:
  * one claim node per (paper, cell); cell = (shift group ID / config /
    visual, fusion naive/regularized, state frame absolute/corrected,
    effect class); config = spatial, embodiment, robot_init or injected
    perturbation; visual = camera fault (robot configuration in
    distribution); corrected = relative, grounded or aligned. Its label
    carries the paper-cell mean effect.
  * claim -derived_from-> paper -derived_from-> provenance roots:
    the backbone family and the benchmark/robot setup. Two papers that
    fine-tune the same base model on the same benchmark share roots,
    so they do not count as independent evidence.
  * claim -supports/contradicts-> hypothesis: each hypothesis predicts
    an effect class per cell (predict()); a cell supports the hypotheses
    whose prediction it matches and contradicts the others, and is only
    linked when the hypotheses disagree about it (rules()).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT.parents[1] / "scientific-evidence-planner"
sys.path.insert(0, str(PLANNER / "src"))

import yaml  # noqa: E402
from planner import (ActionGenerator, BayesianBeliefUpdater, EdgeType,  # noqa: E402
                     EvidenceEdge, EvidenceGraph, EvidenceNode, GapAnalyzer,
                     GapAnalyzerConfig, Hypothesis, NewEvidence, NodeType, Planner,
                     Question, UtilityEstimator, entropy, load_utility_config)

OUTCOMES = ("help", "null", "hurt")

QUESTION = Question(
    question_id="robotics_proprio_001",
    question=("Why do reported effects of proprioceptive-state input on manipulation "
              "policy success conflict in sign (state-free 0->85% vs. state +3 to +30 pp)?"),
    domain="robot_learning",
)
HYPS = {
    "H1_shift": "Sign is set by evaluation shift: state hurts out of the training range, "
                "is neutral-to-helpful in distribution.",
    "H2_frame": "The harm comes from absolute frames (and velocity terms); relative or "
                "grounded state helps under any evaluation.",
    "H3_fusion": "Naive fusion lets state dominate optimisation and suppress vision; with "
                 "regularised/grounded fusion state helps.",
    "H4_observability": "State helps when cameras miss task-relevant information "
                        "(single fixed view, bimanual relations); redundant with full wrist views.",
    "H5_heterogeneous": "No general factor: effects are task- and setup-specific noise.",
    # Added after reading the remaining flagged papers (post hoc, see README):
    "H6_shift_type": "Sign is set by the TYPE of shift: state hurts when the robot/scene "
                     "configuration leaves the training range, helps when the cameras fail.",
    "H7_shift_x_encoding": "State hurts only under configuration shift AND raw (absolute, "
                           "naive) encoding; corrected frame or regularised fusion makes it help "
                           "there; it helps under camera faults; neutral-to-helpful in distribution.",
}


def shift_group(s):
    return "ID" if s == "ID" else "visual" if s == "visual_fault" else "config"


def cell_class(e):
    return "hurt" if e < -3 else "help" if e > 3 else "null"


def predict(h, c):
    """Effect class each hypothesis predicts for a cell (None = no prediction)."""
    if h == "H1_shift":
        return "hurt" if c["shift"] != "ID" else "not_hurt"
    if h == "H2_frame":
        return "help" if c["frame"] == "corrected" else "hurt"
    if h == "H3_fusion":
        return "hurt" if c["fusion"] == "naive" else "help"
    if h == "H4_observability":
        return "help" if (not c["wrist"] or c["shift"] == "visual") else "not_help"
    if h == "H5_heterogeneous":
        return "null"
    if h == "H6_shift_type":
        return {"config": "hurt", "visual": "help"}.get(c["shift"], "not_hurt")
    if h == "H7_shift_x_encoding":
        if c["shift"] == "config":
            return "hurt" if (c["frame"] == "absolute" and c["fusion"] == "naive") else "help"
        return "help" if c["shift"] == "visual" else "not_hurt"
    return None


def matches(pred, k):
    return {"not_hurt": k != "hurt", "not_help": k != "help"}.get(pred, pred == k)


def rules(c):
    """Discriminating evidence only: a cell is linked to the hypotheses
    only when they disagree about it (if every hypothesis predicts the
    observed class, the likelihood ratio is ~1 and it is not evidence).
    Confidence: 0.6 per cell, 0.4 when the cell rests on a single
    setting (one row)."""
    verdicts = {h: matches(predict(h, c), c["cls"]) for h in HYPS}
    if len(set(verdicts.values())) == 1:
        return []
    conf = 0.6 if c["n_rows"] > 1 else 0.4
    return [(h, "supports" if ok else "contradicts", conf) for h, ok in verdicts.items()]


def root_ids(r):
    bb = r["backbone"].split("-")[0].split(" ")[0].lower()
    bench = (r.get("benchmark") or f"{r['paper']}_{r['platform']}_setup").lower().replace(" ", "_")
    return f"root_model_{bb}", f"root_data_{bench}"


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "results.jsonl")]
    # Same unit as single_variable_test.py: with-vs-without-state only.
    # 'terms', 'frame' and 'fusion' rows have a different baseline.
    rows = [r for r in rows if r["effect_pp"] is not None and r["comparison"] == "presence"]
    cells = defaultdict(list)
    for r in rows:
        key = (r["paper"], shift_group(r["shift"]), r["fusion"],
               "absolute" if r["state_frame"] == "absolute" else "corrected", cell_class(r["effect_pp"]))
        cells[key].append(r)

    g = EvidenceGraph()
    cell_info = []
    for hid, desc in HYPS.items():
        g.add_node(EvidenceNode(hid, NodeType.HYPOTHESIS, desc))
    added = set()
    for (paper, shift, fusion, frame, k), rs in cells.items():
        year = str(rs[0]["year"])
        pid = f"paper_{paper}"
        if pid not in added:
            g.add_node(EvidenceNode(pid, NodeType.PAPER, paper, timestamp=year))
            added.add(pid)
        for r in rs:
            for root in root_ids(r):
                if root not in added:
                    g.add_node(EvidenceNode(root, NodeType.DATASET, root))
                    added.add(root)
                if not any(e.target == root for e in g.out_edges(pid, EdgeType.DERIVED_FROM)):
                    g.add_edge(EvidenceEdge(pid, root, EdgeType.DERIVED_FROM))
        mean = sum(r["effect_pp"] for r in rs) / len(rs)
        cid = f"claim_{paper}_{shift}_{fusion}_{frame}_{k}"
        g.add_node(EvidenceNode(cid, NodeType.CLAIM,
                                f"{paper}: {k} ({mean:+.1f} pp, n={len(rs)}), "
                                f"{shift}, {fusion}, {frame}",
                                timestamp=year))
        g.add_edge(EvidenceEdge(cid, pid, EdgeType.DERIVED_FROM))
        c = dict(shift=shift, fusion=fusion, frame=frame, cls=k,
                 velocity=any(r["velocity_terms"] for r in rs),
                 wrist=all(r["wrist_camera"] for r in rs), n_rows=len(rs))
        for hid, pol, conf in rules(c):
            g.add_edge(EvidenceEdge(cid, hid, EdgeType(pol), confidence=conf))
        cell_info.append((cid, c))

    hyps = [Hypothesis(h, d, prior=1 / len(HYPS)) for h, d in HYPS.items()]
    cfg = yaml.safe_load((PLANNER / "configs" / "paper3_informed.yaml").read_text())
    gcfg = cfg["gap_analysis"]
    report = GapAnalyzer(GapAnalyzerConfig(
        min_evidence_count=gcfg["min_evidence_count"],
        min_support_confidence=gcfg["min_support_confidence"],
        require_independent_sources=gcfg["require_independent_sources"])).analyze(QUESTION, hyps, g)
    actions = ActionGenerator().generate(report)
    recs = Planner(UtilityEstimator(config=load_utility_config(PLANNER / "configs" / "paper3_informed.yaml"))
                   ).rank(actions, hyps, g, top_k=cfg["planner"]["top_k"])

    posterior, scorecard = update_beliefs(hyps, cell_info)
    # Robustness: cells from one paper are not independent, so the pooled
    # posterior is overconfident. Leave each paper out and re-run.
    loo = {}
    for paper in sorted({cid.split("_")[1] for cid, _ in cell_info}):
        post, _ = update_beliefs(hyps, [(cid, c) for cid, c in cell_info if cid.split("_")[1] != paper])
        win = max(post, key=post.get)
        loo[paper] = dict(winner=win, p_winner=post[win], p_H1=post["H1_shift"],
                          p_H7=post["H7_shift_x_encoding"])
    experiments = rank_cell_experiments(posterior, cell_info)

    # Per-hypothesis evidence summary (planner graph queries).
    summary = {}
    for h in HYPS:
        sup, con = g.evidence_for(h), g.evidence_against(h)
        roots = set()
        for s in g.evidence_sources(h):
            roots |= g.provenance_roots(s)
        summary[h] = dict(n_support=len(sup), n_contradict=len(con),
                          support_conf=round(sum(e.confidence for e in sup), 2),
                          contradict_conf=round(sum(e.confidence for e in con), 2),
                          n_provenance_roots=len(roots),
                          support_papers=sorted({e.source.split('_')[1] for e in sup}),
                          contradict_papers=sorted({e.source.split('_')[1] for e in con}))

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    g.save(out_dir / "planner_graph.json")
    (out_dir / "planner_output.json").write_text(json.dumps(dict(
        question=QUESTION.to_dict(), hypothesis_evidence=summary,
        gaps=[gap.to_dict() for gap in report.gaps],
        posterior=posterior, scorecard=scorecard, leave_one_paper_out=loo, cell_experiments=experiments,
        recommendations=[r.to_dict() for r in recs]), indent=1))

    print(f"graph: {len(g)} nodes, {len(g.edges())} edges, {len(cells)} paper-cells\n")
    for h, s in summary.items():
        print(f"{h:18s} +{s['n_support']} ({s['support_conf']}) / -{s['n_contradict']} "
              f"({s['contradict_conf']})  roots={s['n_provenance_roots']}")
    print("\ngaps:")
    for gap in report.gaps:
        print(f"  {gap.hypothesis_id}: {gap.reason} (conf {gap.confidence})")
    print("\ntop actions:")
    for r in recs:
        print(f"  {r.rank}. [{r.action.action_type.value}] -> {r.action.target_hypothesis} "
              f"U={r.utility.total:.3f}  {r.reasoning}")
    print("\nposterior after replaying all cells (uniform prior):")
    for h, p in posterior.items():
        sc = scorecard[h]
        print(f"  {h:18s} p={p:.3f}  correct on {sc['correct']}/{sc['decided']} cells")
    print("\nleave-one-paper-out: winner p(winner) / p(H1) / p(H7)")
    for paper, v in loo.items():
        print(f"  drop {paper}: {v['winner']} {v['p_winner']:.3f} / {v['p_H1']:.3f} / {v['p_H7']:.3f}")
    print("\ncell-level experiments by expected information gain:")
    for e in experiments[:8]:
        print(f"  EIG={e['eig_bits']:.3f}  {e['cell']}  existing papers={e['existing_papers']}  "
              f"[{e['action_type']}]")


def likelihood(h, c, observed):
    """P(observed class | h): 0.75 on the predicted class(es), rest spread."""
    pred = predict(h, c)
    allowed = [o for o in OUTCOMES if matches(pred, o)]
    return 0.75 / len(allowed) if observed in allowed else 0.25 / (len(OUTCOMES) - len(allowed))


def update_beliefs(hyps, cell_info):
    """Replay every observed cell through the Paper 4 Bayesian updater."""
    updater = BayesianBeliefUpdater()
    scorecard = {h.hypothesis_id: dict(correct=0, decided=0) for h in hyps}
    for cid, c in cell_info:
        lk = {h.hypothesis_id: likelihood(h.hypothesis_id, c, c["cls"]) for h in hyps}
        if c["n_rows"] == 1:  # temper single-setting cells toward neutral
            lk = {k: 0.5 + 0.5 * v for k, v in lk.items()}
        hyps = updater.update(hyps, NewEvidence(cid, cid, lk))
        for h in scorecard:
            scorecard[h]["decided"] += 1
            scorecard[h]["correct"] += int(matches(predict(h, c), c["cls"]))
    return {h.hypothesis_id: round(h.prior, 4) for h in hyps}, scorecard


def rank_cell_experiments(posterior, cell_info):
    """Preposterior analysis (Paper 4's information-gain term) over design
    cells: which with/without-state comparison would most reduce
    uncertainty about which hypothesis is right?"""
    covered = defaultdict(set)
    for cid, c in cell_info:
        key = (c["shift"], c["frame"], c["fusion"], "wrist" if c["wrist"] else "no-wrist")
        covered[key].add(cid.split("_")[1])
    h_now = entropy(list(posterior.values()))
    out = []
    for shift in ("ID", "config", "visual"):
        for frame in ("absolute", "corrected"):
            for fusion in ("naive", "regularized"):
                for wrist in (True, False):
                    c = dict(shift=shift, frame=frame, fusion=fusion, wrist=wrist)
                    eig = 0.0
                    for o in OUTCOMES:
                        joint = {h: p * likelihood(h, c, o) for h, p in posterior.items()}
                        z = sum(joint.values())
                        eig += z * (h_now - entropy([v / z for v in joint.values()]))
                    key = (shift, frame, fusion, "wrist" if wrist else "no-wrist")
                    out.append(dict(cell=" / ".join(key), eig_bits=round(eig, 4),
                                    existing_papers=sorted(covered.get(key, set())),
                                    action_type="rerun_analysis (released checkpoints, sim)"
                                    if shift == "ID" else "run_simulation + collect_observation (real shift)"))
    return sorted(out, key=lambda e: (-e["eig_bits"], len(e["existing_papers"])))


if __name__ == "__main__":
    main()
