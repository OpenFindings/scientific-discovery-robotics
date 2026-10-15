"""Pre-registered map from pilot outcomes to planner posteriors.

Before any pilot run, enumerate every combination of effect classes
(help / null / hurt) for the four shifted pilot cells and replay each
through the same likelihood the evidence map uses
(evidence_map/scripts/run_planner.py), starting from the map's current
posterior. The in-distribution cells are held at 'null' (the map's
working assumption); they discriminate little between H6 and H7.

Pilot cells (pi0.5, scene + wrist camera, naive fusion):
  RK  raw absolute state  vs none, configuration shift
  CK  episode-relative    vs none, configuration shift
  RF  raw absolute state  vs none, camera fault
  CF  episode-relative    vs none, camera fault

Writes pilot/results/preregistered_outcomes.json.
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evidence_map" / "scripts"))
import run_planner as rp  # noqa: E402

CELLS = {
    "RK": dict(shift="config", frame="absolute", fusion="naive", wrist=True),
    "CK": dict(shift="config", frame="corrected", fusion="naive", wrist=True),
    "RF": dict(shift="visual", frame="absolute", fusion="naive", wrist=True),
    "CF": dict(shift="visual", frame="corrected", fusion="naive", wrist=True),
    "RID": dict(shift="ID", frame="absolute", fusion="naive", wrist=True),
    "CID": dict(shift="ID", frame="corrected", fusion="naive", wrist=True),
}


def update(post, obs):
    post = dict(post)
    for cell, o in obs.items():
        post = {h: p * rp.likelihood(h, CELLS[cell], o) for h, p in post.items()}
        z = sum(post.values())
        post = {h: p / z for h, p in post.items()}
    return post


def main():
    prior = json.loads((ROOT / "evidence_map" / "results" / "planner_output.json").read_text())["posterior"]
    rows = []
    for combo in itertools.product(rp.OUTCOMES, repeat=4):
        obs = dict(zip(("RK", "CK", "RF", "CF"), combo), RID="null", CID="null")
        # prior predictive probability of this outcome pattern
        pred = sum(p * _prod(rp.likelihood(h, CELLS[c], o) for c, o in obs.items())
                   for h, p in prior.items())
        post = update(prior, obs)
        win = max(post, key=post.get)
        rows.append(dict(RK=combo[0], CK=combo[1], RF=combo[2], CF=combo[3],
                         prior_predictive=round(pred, 5), winner=win, p_winner=round(post[win], 3),
                         p_H1=round(post["H1_shift"], 3), p_H6=round(post["H6_shift_type"], 3),
                         p_H7=round(post["H7_shift_x_encoding"], 3)))
    z = sum(r["prior_predictive"] for r in rows)
    for r in rows:
        r["prior_predictive"] = round(r["prior_predictive"] / z, 4)
    rows.sort(key=lambda r: -r["prior_predictive"])
    out = ROOT / "pilot" / "results"
    out.mkdir(exist_ok=True)
    (out / "preregistered_outcomes.json").write_text(json.dumps(dict(prior=prior, outcomes=rows), indent=1))
    print("current posterior:", {h: round(p, 3) for h, p in prior.items()})
    print(f"\n{'RK':5s}{'CK':5s}{'RF':5s}{'CF':5s} P(pattern)  winner              p(H1)  p(H6)  p(H7)")
    for r in rows[:14]:
        print(f"{r['RK']:5s}{r['CK']:5s}{r['RF']:5s}{r['CF']:5s} {r['prior_predictive']:.3f}      "
              f"{r['winner']:19s} {r['p_H1']:.3f}  {r['p_H6']:.3f}  {r['p_H7']:.3f}")
    # expected information gain of the whole pilot
    h0 = rp.entropy(list(prior.values()))
    eig = 0.0
    for r in rows:
        post = update(prior, dict(RK=r["RK"], CK=r["CK"], RF=r["RF"], CF=r["CF"], RID="null", CID="null"))
        eig += r["prior_predictive"] * (h0 - rp.entropy(list(post.values())))
    print(f"\nexpected information gain of the pilot: {eig:.3f} bits (prior entropy {h0:.3f})")


def _prod(xs):
    out = 1.0
    for x in xs:
        out *= x
    return out


if __name__ == "__main__":
    main()
