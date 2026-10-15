"""Deduplicate the citation pool and flag candidates for manual reading.

The regex screen is deliberately loose (high recall); it only decides
which citation-retrieved papers get read. Inclusion in the evidence map
is a manual decision recorded in ../data/screening.jsonl.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pool = json.loads((ROOT / "data" / "citation_pool.json").read_text())
papers = {}
for key, rows in pool.items():
    for p in rows:
        pid = (p.get("externalIds") or {}).get("ArXiv") or p["title"].lower()
        papers.setdefault(pid, {**p, "via": []})["via"].append(key)

STATE = re.compile(r"propriocept|robot state|state input|state[- ]free|without state|copycat|"
                   r"causal confusion|shortcut|end[- ]effector pose|joint state|state-based", re.I)
ROBOT = re.compile(r"robot|manipulat|visuomotor|imitation|behavior(al)? cloning|policy", re.I)
DRIVING = re.compile(r"driving|vehicle|planner|navigation|dialogue", re.I)

flagged = []
for pid, p in papers.items():
    text = f"{p['title']} {p.get('abstract') or ''}"
    if STATE.search(text) and ROBOT.search(text) and not DRIVING.search(p["title"]):
        flagged.append({"id": pid, "year": p.get("year"), "title": p["title"], "n_paths": len(p["via"])})
flagged.sort(key=lambda r: (r["year"] or 0, r["id"]))
(ROOT / "data" / "pool_dedup.json").write_text(json.dumps(papers, indent=1))
(ROOT / "data" / "screen_flagged.json").write_text(json.dumps(flagged, indent=1))
print(len(papers), "unique papers;", len(flagged), "flagged")
