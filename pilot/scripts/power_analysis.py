"""Power / seed-count analysis for the 3x2 LIBERO-Plus pilot.

Unit of inference is the trained policy (arm x seed), not the episode:
with a few hundred paired episodes per condition the binomial error is
~1-2 pp, so between-seed variation dominates. Each arm contrast is
tested with a two-sample t-test on seed-level success rates
(df = 2k - 2), at alpha = 0.05 and at the Holm-strictest level for the
four primary contrasts (0.05 / 4).

Monte Carlo: seed-level success = p_arm + N(0, sd_seed) + binomial noise
over n_eps episodes (pairing across arms ignored, so this is
conservative). Writes pilot/results/power.json and prints the table.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def t_cdf(t, df, n=4000):
    """Student-t CDF by Simpson integration of the density (no scipy)."""
    if t < 0:
        return 1 - t_cdf(-t, df, n)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    xs = np.linspace(0, t, n + 1)
    ys = c * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
    h = t / n
    return 0.5 + h / 3 * (ys[0] + ys[-1] + 4 * ys[1:-1:2].sum() + 2 * ys[2:-1:2].sum())


def t_crit(alpha, df):
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 1 - t_cdf(mid, df) > alpha / 2 else (lo, mid)
    return (lo + hi) / 2


def power(delta, p0, sd_seed, k, n_eps, alpha, sims=20000):
    df = 2 * k - 2
    tc = t_crit(alpha, df)
    pa, pb = p0, p0 + delta
    sa = np.clip(pa + RNG.normal(0, sd_seed, (sims, k)), 0, 1)
    sb = np.clip(pb + RNG.normal(0, sd_seed, (sims, k)), 0, 1)
    ya = RNG.binomial(n_eps, sa) / n_eps
    yb = RNG.binomial(n_eps, sb) / n_eps
    sp = np.sqrt((ya.var(axis=1, ddof=1) + yb.var(axis=1, ddof=1)) / 2)
    t = (yb.mean(1) - ya.mean(1)) / (sp * np.sqrt(2 / k) + 1e-12)
    return float((np.abs(t) > tc).mean())


def main():
    out = []
    for n_eps in (400, 1200):
        for sd in (0.02, 0.04, 0.06, 0.08):
            for k in (2, 3, 4, 5):
                row = dict(n_eps=n_eps, sd_seed_pp=round(sd * 100), seeds_per_arm=k)
                for d in (0.10, 0.15, 0.20):
                    row[f"power_d{int(d*100)}_a05"] = power(d, 0.40, sd, k, n_eps, 0.05)
                    row[f"power_d{int(d*100)}_holm"] = power(d, 0.40, sd, k, n_eps, 0.0125)
                out.append(row)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "power.json").write_text(json.dumps(out, indent=1))
    print("baseline success 40%; power at alpha=.05 / Holm (.0125)")
    print("n_eps sd_seed k |  d=10pp      d=15pp      d=20pp")
    for r in out:
        print(f"{r['n_eps']:5d} {r['sd_seed_pp']:5d}pp {r['seeds_per_arm']} | "
              + "  ".join(f"{r[f'power_d{d}_a05']:.2f}/{r[f'power_d{d}_holm']:.2f}" for d in (10, 15, 20)))


if __name__ == "__main__":
    main()
