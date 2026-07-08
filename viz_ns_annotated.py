#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_ns_annotations.py
=====================
Render a sentence-level NS annotation file (ns_sentences_annotated*.jsonl) as ONE
self-contained interactive HTML page for manual annotation audit.

Per story (whole story as continuous text, not sentence boxes):
  * mentions highlighted, color = coreference cluster (golden-angle hues; larger
    chains get the most distinct colors); singletons = warm gray, dashed underline
  * NP-layer mentions (source: "np_layer", if present in the file) = dotted outline
  * nominal tokens covered by NO mention = red dotted "query" underline (candidate
    missed entities -- the thing being audited)
  * hover tooltip: cluster id, chain size, role, NER, given/new at that sentence
  * click a mention or a ledger row -> focus that chain (everything else fades);
    click again or press Esc to clear
  * chain ledger per story: color chip, representative head, mention count, span
    of sentences; sorted by chain size
  * toggles: singletons, uncovered marks, sentence indices

Usage:  python viz_ns_annotations.py --in ns_sentences_annotated_litbank.jsonl \
            --out ns_annotation_audit.html
"""

import argparse
import html
import json
from collections import Counter, defaultdict

NOMINAL = ("NN", "NNS", "NNP", "NNPS", "PRP", "PRP$")
_PRON = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its",
         "their", "i", "you", "we", "me", "us", "my", "your", "our", "this",
         "that", "these", "those"}


def rep_head(texts):
    heads = Counter()
    for t in texts:
        core = t.lower().split(" of ")[0].strip('.,";\u201d\u2019 ')
        w = core.split()[-1] if core.split() else ""
        if w and w not in _PRON:
            heads[w] += 1
    return heads.most_common(1)[0][0] if heads else (texts[0][:18] if texts else "?")


def render_sentence(rec, colors, show_sid=True):
    """Nested-span HTML for one sentence: mentions + uncovered nominal marks."""
    text = rec["target"]
    toks = rec["segments"]["target"]["tokens"]
    given = set(rec.get("given_entities") or [])
    lemma_given = set(rec.get("lemma_given_entities") or [])

    spans = []
    m_spans = [(m["char_start"], m["char_end"]) for m in rec["target_mentions"]]
    for m in rec["target_mentions"]:
        cid = m["cluster_id"]
        cls = ["m"]
        if m.get("singleton") or cid not in colors:
            cls.append("sing")
        if m.get("source") == "np_layer":
            cls.append("npl")
        status = ("given" if cid in given else
                  "lemma-given" if cid in lemma_given else "new")
        tip = (f"cluster {cid} \u00b7 {colors.get(cid, {}).get('n', 1)} mention(s) "
               f"\u00b7 {m.get('grammatical_role') or '?'}"
               + (f" \u00b7 {m['ner_label']}" if m.get("ner_label") else "")
               + f" \u00b7 {status}")
        style = ""
        if cid in colors:
            h = colors[cid]["h"]
            style = (f"--h:{h}")
        spans.append((m["char_start"], m["char_end"], 0,
                      f'<span class="{" ".join(cls)}" data-cid="c{cid}" '
                      f'style="{style}" data-tip="{html.escape(tip, quote=True)}">',
                      "</span>"))
    for t in toks:
        if t.get("pos") in NOMINAL and t["char_start"] is not None:
            if not any(t["char_start"] < e and t["char_end"] > s for s, e in m_spans):
                spans.append((t["char_start"], t["char_end"], 1,
                              '<span class="unc" data-tip="nominal token in no '
                              'mention (candidate miss)">', "</span>"))

    # nested rendering; partial overlaps: inner conflicting span is dropped
    spans.sort(key=lambda s: (s[0], -s[1], s[2]))
    out, stack, pos, dropped = [], [], 0, 0
    for s, e, _, open_tag, close_tag in spans:
        while stack and stack[-1][1] <= s:
            ce = stack.pop()[1]
            out.append(html.escape(text[pos:ce])); out.append("</span>")
            pos = ce
        if stack and e > stack[-1][1]:
            dropped += 1
            continue
        out.append(html.escape(text[pos:s])); out.append(open_tag)
        pos = s
        stack.append((s, e))
    while stack:
        ce = stack.pop()[1]
        out.append(html.escape(text[pos:ce])); out.append("</span>")
        pos = ce
    out.append(html.escape(text[pos:]))
    sid = (f'<sup class="sid">{rec["sentence_index"]}</sup> ' if show_sid else "")
    return sid + "".join(out), dropped


def build(inp, out):
    recs = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    docs = defaultdict(list)
    for r in recs:
        docs[r["doc_id"]].append(r)
    for d in docs.values():
        d.sort(key=lambda r: r["sentence_index"])

    sections, tabs = [], []
    total_dropped = 0
    for di, (doc_id, drecs) in enumerate(
            sorted(docs.items(), key=lambda kv: int(kv[0].split("_")[-1]))):
        # cluster inventory for this story
        cl_ms = defaultdict(list)
        for r in drecs:
            for m in r["target_mentions"]:
                cl_ms[m["cluster_id"]].append((r["sentence_index"], m["text"]))
        chains = {c: v for c, v in cl_ms.items() if len(v) > 1}
        singles = {c: v for c, v in cl_ms.items() if len(v) == 1}
        order = sorted(chains, key=lambda c: -len(chains[c]))
        colors = {c: {"h": round((i * 137.508) % 360, 1), "n": len(chains[c])}
                  for i, c in enumerate(order)}
        for c in singles:
            colors.setdefault(c, {"h": None, "n": 1})

        # text body
        body, n_unc = [], 0
        for r in drecs:
            frag, dropped = render_sentence(r, colors)
            total_dropped += dropped
            body.append(frag)
            spans = [(m["char_start"], m["char_end"]) for m in r["target_mentions"]]
            for t in r["segments"]["target"]["tokens"]:
                if t.get("pos") in NOMINAL and t["char_start"] is not None and \
                   not any(t["char_start"] < e and t["char_end"] > s for s, e in spans):
                    n_unc += 1
        text_html = " ".join(body)

        # ledger
        rows = []
        for c in order:
            sents = [si for si, _ in chains[c]]
            rows.append(
                f'<button class="row" data-cid="c{c}" style="--h:{colors[c]["h"]}">'
                f'<span class="chip"></span><span class="rh">{html.escape(rep_head([t for _, t in chains[c]]))}</span>'
                f'<span class="ct">{len(chains[c])}\u00d7</span>'
                f'<span class="rng">s{min(sents)}\u2013s{max(sents)}</span></button>')
        ledger = "".join(rows) or '<p class="none">no multi-mention chains</p>'
        name = doc_id.replace("naturalstories_", "Story ")
        stats = (f'{len(chains)} chains \u00b7 {len(singles)} singletons \u00b7 '
                 f'{n_unc} unmarked nominals')
        tabs.append(f'<button class="tab" data-story="st{di}">{name}</button>')
        sections.append(f"""
<section class="story" id="st{di}">
  <div class="reader"><p class="stats">{stats}</p><div class="text">{text_html}</div></div>
  <aside class="ledger"><h2>Chains</h2>{ledger}
    <h2>Key</h2>
    <p class="key"><span class="m demo" style="--h:210">chain mention</span>
    <span class="m sing demo">singleton</span>
    <span class="unc demo">unmarked nominal</span></p>
  </aside>
</section>""")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Natural Stories \u00b7 coreference audit</title>
<style>
:root {{ --paper:#FAF8F3; --ink:#26221B; --chrome:#8A8377; --line:#E4DFD4;
        --query:#B3372B; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--paper); color:var(--ink);
       font:15px/1.45 ui-sans-serif, system-ui, sans-serif }}
header {{ position:sticky; top:0; z-index:5; background:var(--paper);
          border-bottom:1px solid var(--line); padding:10px 22px;
          display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center }}
h1 {{ font:600 15px/1 ui-sans-serif,system-ui; margin:0; letter-spacing:.02em }}
h1 small {{ color:var(--chrome); font-weight:400 }}
.tab {{ border:1px solid var(--line); background:transparent; color:var(--ink);
        padding:4px 10px; border-radius:99px; cursor:pointer; font:inherit }}
.tab.on {{ background:var(--ink); color:var(--paper); border-color:var(--ink) }}
.tab:focus-visible, .row:focus-visible, .tgl input:focus-visible
  {{ outline:2px solid var(--query); outline-offset:2px }}
.tgl {{ color:var(--chrome); font-size:13px; display:flex; gap:4px;
        align-items:center; cursor:pointer }}
.story {{ display:none; max-width:1120px; margin:0 auto; padding:26px 22px;
          gap:34px }}
.story.on {{ display:flex }}
.reader {{ flex:1 1 auto; min-width:0 }}
.stats {{ color:var(--chrome); font-size:13px; margin:0 0 14px }}
.text {{ font:17.5px/1.9 Charter, Georgia, 'Times New Roman', serif;
         max-width:68ch }}
.sid {{ color:var(--chrome); font:10px/1 ui-sans-serif; user-select:none }}
body.no-sid .sid {{ display:none }}
.m {{ background:hsl(var(--h) 65% 88%); border-bottom:2px solid hsl(var(--h) 60% 46%);
      border-radius:2px; padding:0 1px; cursor:pointer }}
.m .m {{ padding:0 }} /* nested mentions stay readable */
.m.sing {{ background:#EFEBE1; border-bottom:2px dashed #A69F8E }}
.m.npl {{ outline:1.5px dotted #A69F8E; outline-offset:1px }}
body.no-sing .m.sing {{ background:transparent; border-bottom:none; outline:none;
                        cursor:text }}
.unc {{ border-bottom:2px dotted var(--query); cursor:help }}
body.no-unc .unc {{ border-bottom:none; cursor:text }}
body.focus .m:not(.hit) {{ background:transparent; border-bottom-color:transparent;
                           opacity:.45 }}
body.focus .m.hit {{ background:hsl(var(--h) 70% 80%);
                     border-bottom:3px solid hsl(var(--h) 65% 40%) }}
body.focus .text {{ color:#6d675c }}
.ledger {{ flex:0 0 240px; position:sticky; top:64px; align-self:flex-start;
           max-height:calc(100vh - 90px); overflow:auto; padding-right:4px }}
.ledger h2 {{ font:600 11px/1 ui-sans-serif; letter-spacing:.14em;
              text-transform:uppercase; color:var(--chrome); margin:18px 0 8px }}
.row {{ display:flex; gap:8px; align-items:baseline; width:100%; text-align:left;
        border:0; background:transparent; font:13px/1.5 ui-sans-serif;
        color:var(--ink); padding:3px 4px; border-radius:6px; cursor:pointer }}
.row:hover {{ background:#F0ECE2 }}
.row.on {{ background:#EAE4D6 }}
.chip {{ width:11px; height:11px; border-radius:3px; flex:none; align-self:center;
         background:hsl(var(--h) 65% 60%) }}
.rh {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap }}
.ct {{ color:var(--chrome) }} .rng {{ color:var(--chrome); font-size:11px }}
.key .demo {{ margin-right:12px; font:14px/1.4 Charter, Georgia, serif }}
.none {{ color:var(--chrome); font-size:13px }}
#tip {{ position:fixed; z-index:9; background:var(--ink); color:var(--paper);
        font:12px/1.4 ui-sans-serif; padding:5px 8px; border-radius:6px;
        pointer-events:none; max-width:280px; display:none }}
@media (max-width:840px) {{ .story{{flex-direction:column}} .ledger{{position:static;
  flex-basis:auto; max-height:none}} }}
@media (prefers-reduced-motion:no-preference) {{ .m{{transition:background .12s,
  opacity .12s}} }}
</style></head><body class="">
<header>
  <h1>Natural Stories <small>coreference audit \u00b7 {html.escape(inp.split('/')[-1])}</small></h1>
  {"".join(tabs)}
  <label class="tgl"><input type="checkbox" id="tg-sing" checked> singletons</label>
  <label class="tgl"><input type="checkbox" id="tg-unc" checked> unmarked nominals</label>
  <label class="tgl"><input type="checkbox" id="tg-sid" checked> sentence n\u00ba</label>
</header>
{"".join(sections)}
<div id="tip"></div>
<script>
const tabs=[...document.querySelectorAll('.tab')],
      stories=[...document.querySelectorAll('.story')], body=document.body;
function show(i){{tabs.forEach((t,j)=>t.classList.toggle('on',i===j));
  stories.forEach((s,j)=>s.classList.toggle('on',i===j)); clearFocus();}}
tabs.forEach((t,i)=>t.onclick=()=>show(i)); show(0);
function clearFocus(){{body.classList.remove('focus');
  document.querySelectorAll('.hit,.row.on').forEach(e=>e.classList.remove('hit','on'));}}
function focusChain(cid){{const already=body.classList.contains('focus')&&
    document.querySelector('.row.on')?.dataset.cid===cid;
  clearFocus(); if(already)return; body.classList.add('focus');
  document.querySelectorAll(`[data-cid="${{cid}}"]`).forEach(e=>
    e.classList.add(e.classList.contains('row')?'on':'hit'));}}
document.addEventListener('click',e=>{{
  const el=e.target.closest('[data-cid]');
  if(el)focusChain(el.dataset.cid); else if(!e.target.closest('.ledger'))clearFocus();}});
document.addEventListener('keydown',e=>{{if(e.key==='Escape')clearFocus();}});
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{{
  const el=e.target.closest('[data-tip]');
  if(el&&!body.classList.contains('no-unc')||el&&!el.classList.contains('unc')){{
    tip.textContent=el.dataset.tip; tip.style.display='block';
    tip.style.left=Math.min(e.clientX+14,innerWidth-290)+'px';
    tip.style.top=(e.clientY+16)+'px';}} else tip.style.display='none';}});
[['tg-sing','no-sing'],['tg-unc','no-unc'],['tg-sid','no-sid']].forEach(([id,cls])=>{{
  document.getElementById(id).onchange=e=>body.classList.toggle(cls,!e.target.checked);}});
</script></body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[done] {len(docs)} stories -> {out}"
          + (f" ({total_dropped} partially-overlapping spans dropped)"
             if total_dropped else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    build(a.inp, a.out)