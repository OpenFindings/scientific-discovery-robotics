#!/bin/bash
# research/paper_draft.md -> publish/paper.pdf (pandoc HTML + headless Chromium print)
set -e
R=$(cd "$(dirname "$0")/.." && pwd); OUT=$R/publish; TMP=$(mktemp -d)
# drop the first H1 and the italic preamble line; the title block below replaces them
awk 'NR==1 && /^# /{next} {print}' $R/research/paper_draft.md | sed '1{/^$/d}' > $TMP/body.md
cat > $TMP/head.html <<'H'
<h1>Whether proprioception hurts a manipulation policy depends on what the policy can see</h1>
<p class="authors">Ada Mao (Independent) · Claude Code (AI agent, Anthropic)</p>
<p class="meta">Preprint · OpenFindings or:2026:39f390f2 · Code and data: github.com/OpenFindings/scientific-discovery-robotics (branch claude/friendly-archimedes-4ft6w0) · CC-BY-4.0</p>
H
pandoc $TMP/body.md -f gfm -t html5 -s --metadata pagetitle="Proprioception and what the policy can see" --css $OUT/paper.css -B $TMP/head.html -o $TMP/paper.html --embed-resources 2>/dev/null || \
pandoc $TMP/body.md -f gfm -t html5 -s --metadata pagetitle="Proprioception and what the policy can see" -H <(echo "<style>$(cat $OUT/paper.css)</style>") -B $TMP/head.html -o $TMP/paper.html
NODE_PATH=$(npm root -g) node -e "
const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://$TMP/paper.html'); await p.pdf({ path: '$OUT/paper.pdf', format: 'A4', printBackground: true,
  displayHeaderFooter: true, headerTemplate: '<span></span>', footerTemplate: '<div style=\"font-size:8px;width:100%;text-align:center;color:#666\"><span class=\"pageNumber\"></span> / <span class=\"totalPages\"></span></div>',
  margin: { top: '16mm', bottom: '16mm', left: '15mm', right: '15mm' } }); await b.close(); })();"
ls -la $OUT/paper.pdf
