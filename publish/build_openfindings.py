"""Build the OpenFindings Research Object (ResearchObjectInput, openfindings.org/openapi.json) from research/paper_draft.md.
Nothing is submitted. Fill the AUTHOR fields, check the code URL is public, then submit via openfindings.org/submit
or POST /api/v1/papers as a verified agent."""
import json, pathlib, re
root = pathlib.Path(__file__).resolve().parents[1]
paper = (root / "research/paper_draft.md").read_text()
abstract = re.search(r"## Abstract\n\n(.*?)\n---", paper, re.S).group(1).strip()
body = paper.split("---", 1)[1].strip()
ro = {
    "title": "Whether proprioception hurts a manipulation policy depends on what the policy can see",
    "abstract": abstract,
    "subject": "Robotics",
    "authors": [{"name": "Ada Mao", "affiliation": "Independent"}, {"name": "Claude Code", "affiliation": "Anthropic (AI agent)"}],
    "provenance": {
        "origin": "supervised",
        "operator": "Ada Mao",
        "agent_id": "840d49d8-196a-42bf-9617-eb149cc9ecee",
        "model": "Claude (Anthropic), via Claude Code",
        "notes": "Literature coding, simulation code, experiments and draft were produced by an AI agent under the direction of Ada Mao, "
                 "who chose the questions and reviewed the outputs. No independent human second coder for the meta-analysis.",
    },
    "body_markdown": body,
    "keywords": ["proprioception", "robot manipulation", "vision-language-action models", "LIBERO", "distribution shift",
                 "shortcut learning", "meta-analysis", "testable predictions"],
    "license": "CC-BY-4.0",
    "code": [{"label": "Code, per-episode results and analysis", "url": "https://github.com/OpenFindings/scientific-discovery-robotics/tree/claude/friendly-archimedes-4ft6w0"}],
    "references": [{"id": i} for i in ["arXiv:2509.18644", "arXiv:2609.39145", "arXiv:2608.03052", "arXiv:2510.03827", "arXiv:2609.03715",
                                        "arXiv:2605.13067", "arXiv:2609.09213", "arXiv:2511.00555", "arXiv:2402.19249", "arXiv:1905.11979",
                                        "arXiv:2010.14876", "arXiv:2504.16054"]],
}
out = root / "publish/openfindings_research_object.json"
out.write_text(json.dumps(ro, indent=1, ensure_ascii=False))
print(out, "abstract", len(abstract), "body", len(body), "chars")
