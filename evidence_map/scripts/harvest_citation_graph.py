"""Harvest the citation neighbourhood of the proprioceptive-state cluster.

Retrieval follows the citation graph (Paper 4's lesson): for every seed
paper we collect (a) papers that cite it and (b) papers it cites, from
Semantic Scholar. The union is the candidate pool; screening for actual
with/without-state results is a separate, manual step
(see ../data/screening.jsonl and ../data/results.jsonl).

Usage:  python evidence_map/scripts/harvest_citation_graph.py
Writes evidence_map/data/citation_pool.json incrementally (resumable).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "citation_pool.json"

# Seeds: the three 2025-26 papers whose results conflict, plus the two
# papers that introduced the state-shortcut ("copycat") hypothesis.
SEEDS = {
    "arXiv:2509.18644": "Do You Need Proprioceptive States in Visuomotor Policies?",
    "arXiv:2608.03052": "How Should VLA Models Use Proprioceptive State?",
    "arXiv:2609.02546": "ZETA: zero-shot cross-embodiment VLA transfer",
    "arXiv:1905.11979": "Causal Confusion in Imitation Learning",
    "arXiv:2010.14876": "Fighting Copycat Agents in Behavioral Cloning from Observation Histories",
    # Second hop: first-hop hits that themselves run state ablations.
    "arXiv:2602.12032": "When would Vision-Proprioception Policies Fail?",
    "arXiv:2605.13067": "When Absolute State Fails",
    "arXiv:2607.07101": "GeoProp",
    "arXiv:2601.16667": "ReViP",
    "arXiv:2506.13679": "ROSA",
}
FIELDS = "title,abstract,year,publicationDate,externalIds,venue"
API = "https://api.semanticscholar.org/graph/v1/paper/{pid}/{rel}?fields={f}&limit=100&offset={o}"


def get(url: str) -> dict | None:
    for attempt in range(40):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "robotics-evidence-map/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2.0 if e.code == 429 else 3.0 * min(attempt + 1, 5))
        except Exception:
            time.sleep(3.0)
    return None


def neighbours(pid: str, rel: str, max_pages: int = 10) -> list[dict]:
    key = "citingPaper" if rel == "citations" else "citedPaper"
    out = []
    for page in range(max_pages):
        payload = get(API.format(pid=pid, rel=rel, f=FIELDS, o=page * 100))
        time.sleep(1.1)
        if not payload or not payload.get("data"):
            break
        out += [row[key] for row in payload["data"] if row.get(key, {}).get("title")]
        if len(payload["data"]) < 100:
            break
    return out


def main() -> None:
    pool = json.loads(OUT.read_text()) if OUT.exists() else {}
    for pid, name in SEEDS.items():
        for rel in ("citations", "references"):
            key = f"{pid}|{rel}"
            if key in pool:
                continue
            rows = neighbours(pid, rel)
            pool[key] = rows
            OUT.write_text(json.dumps(pool, indent=1))
            print(f"{name[:50]:50s} {rel:10s} {len(rows)}")


if __name__ == "__main__":
    main()
