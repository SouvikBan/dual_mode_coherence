"""
prepare_naturalstories.py
=========================
Turn the Natural Stories corpus (https://github.com/languagemit/naturalstories)
into a story-level JSONL that `annotate_naturalstories.py` consumes.

Natural Stories is continuous narrative with self-paced reading times, not
(context, target) acceptability pairs. The natural unit for an information value
analysis is the *sentence*: target = sentence, context = preceding discourse, and
the psychometric measure is the summed word reading time over the sentence (as
Giulianelli et al. 2023 treat PROVO / BROWN).

This script is model-free. It:
  * reconstructs each story's text from the SPR tokenization (punctuation is
    glued to each token in all_stories.tok, so " ".join gives clean text),
  * attaches the aggregated per-token reading times (merged on (item, zone) --
    NOT wordform, since some SPR wordforms contain typos), and
  * embeds the corpus's GOLD Universal Dependencies parse (parses/ud/
    stories-aligned.conllx): sentence segmentation + per-token gold dep/head/POS,
    character-aligned to the reconstructed text via the TokenId={item}.{zone}
    codes. Sentence boundaries and grammatical roles therefore come from the gold
    UD parse, not from a re-parse.

Inputs (inside the cloned repo, overridable)
--------------------------------------------
  naturalstories_RTS/all_stories.tok
  naturalstories_RTS/processed_wordinfo.tsv
  parses/ud/stories-aligned.conllx

Output: one JSON object per story
---------------------------------
{
  "doc_id": "naturalstories_1", "dataset": "naturalstories", "item": 1,
  "text": "<full reconstructed story>",
  "n_tokens": 1073,
  "tokens":      [ {zone, word, char_start, char_end, mean_rt, gmean_rt, sd_rt, n_subj}, ... ],  # SPR units (carry RT)
  "ud_tokens":   [ {g, form, pos, dep, head_g, char_start, char_end, zone, sentence_index}, ... ], # gold UD (carry syntax)
  "ud_sentences":[ {sentence_index, char_start, char_end, zones:[...], token_gids:[...]}, ... ]
}
  - SPR `tokens` carry the reading times (one per zone).
  - `ud_tokens` carry the gold grammatical roles; `head_g` is a 0-based index into
    `ud_tokens` (-1 for the sentence root). A zone may map to several UD tokens
    (e.g. "England," -> "England", ",").

Usage
-----
  python prepare_naturalstories.py --repo /path/to/naturalstories --out naturalstories.jsonl
  # or point at files individually with --tok / --wordinfo / --ud-conll
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

TOKENID_RE = re.compile(r"TokenId=([0-9]+)\.([0-9]+)")

# PTB normalises quotes/brackets; Natural Stories prose uses single quotes for dialogue.
def _candidates(form):
    if form in {"``", "''", "`"}:
        return ["'", '"', "`"]
    if form == "'":
        return ["'", "\u2019", "`"]
    if form == '"':
        return ['"', "\u201c", "\u201d"]
    if form == "-LRB-":
        return ["("]
    if form == "-RRB-":
        return [")"]
    if form == "--":
        return ["--", "\u2014", "\u2013"]
    return [form]


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def load_rt_index(path):
    idx = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            idx[(_to_int(r["item"]), _to_int(r["zone"]))] = {
                "mean_rt": _to_float(r.get("meanItemRT")),
                "gmean_rt": _to_float(r.get("gmeanItemRT")),
                "sd_rt": _to_float(r.get("sdItemRT")),
                "n_subj": _to_int(r.get("nItem")),
            }
    return idx


def load_tokens(path):
    by_item = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by_item[_to_int(r["item"])].append({"zone": _to_int(r["zone"]), "word": r["word"]})
    for item in by_item:
        by_item[item].sort(key=lambda d: d["zone"])
    return by_item


def load_ud(path):
    """item -> list of sentences; each sentence -> list of {form,pos,dep,head,zone} (head 1-based, 0=root)."""
    sents, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 10:
                continue
            m = TOKENID_RE.search(c[9])
            item = int(m.group(1)) if m else None
            zone = int(m.group(2)) if m else None
            cur.append({"id": int(c[0]), "form": c[1], "pos": c[4],
                        "head": int(c[6]), "dep": c[7], "item": item, "zone": zone})
        if cur:
            sents.append(cur)
    by_item = defaultdict(list)
    for s in sents:
        items = {t["item"] for t in s if t["item"] is not None}
        if items:
            by_item[items.pop()].append(s)
    return by_item


def align_ud_to_chars(text, zone_spans, ud_sentence_list):
    """Assign each UD token a character span in `text`.
    zone_spans: {zone: (char_start, char_end)} for SPR tokens.
    Returns flat ud_tokens (global 0-based, head_g resolved) and ud_sentences."""
    ud_tokens, ud_sentences = [], []
    g = 0
    for si, sent in enumerate(ud_sentence_list):
        gid_start = g
        # group this sentence's UD tokens by zone to align within each zone span
        by_zone = defaultdict(list)
        order = []
        for t in sent:
            if t["zone"] not in by_zone:
                order.append(t["zone"])
            by_zone[t["zone"]].append(t)
        sent_gids = []
        for zone in order:
            cs, ce = zone_spans.get(zone, (None, None))
            cur = cs if cs is not None else 0
            for t in by_zone[zone]:
                tok_cs, tok_ce = None, None
                if cs is not None:
                    hit = -1
                    clen = 0
                    for cand in _candidates(t["form"]):
                        p = text.find(cand, cur, ce)
                        if p >= 0:
                            hit, clen = p, len(cand)
                            break
                    if hit < 0:  # typo / unmatchable: fall back to length-based placement
                        hit = cur
                        clen = min(len(t["form"]), max(0, ce - cur))
                    tok_cs, tok_ce = hit, hit + clen
                    cur = tok_ce
                ud_tokens.append({
                    "g": g, "form": t["form"], "pos": t["pos"], "dep": t["dep"],
                    "head_local": t["head"], "_sent_start_g": gid_start,
                    "char_start": tok_cs, "char_end": tok_ce,
                    "zone": zone, "sentence_index": si,
                })
                sent_gids.append(g)
                g += 1
        # sentence char span from its tokens (fallback to zone spans)
        spans = [(ud_tokens[k]["char_start"], ud_tokens[k]["char_end"])
                 for k in sent_gids if ud_tokens[k]["char_start"] is not None]
        s_cs = min(s for s, _ in spans) if spans else None
        s_ce = max(e for _, e in spans) if spans else None
        zones_in = []
        for z in order:
            if z not in zones_in:
                zones_in.append(z)
        ud_sentences.append({"sentence_index": si, "char_start": s_cs, "char_end": s_ce,
                             "zones": zones_in, "token_gids": sent_gids})
    # resolve head_local (1-based within sentence, 0=root) -> head_g (global, -1 root)
    for t in ud_tokens:
        h = t.pop("head_local")
        sg = t.pop("_sent_start_g")
        t["head_g"] = -1 if h == 0 else sg + (h - 1)
    return ud_tokens, ud_sentences


def build(repo, out, tok_path=None, wordinfo_path=None, ud_path=None):
    repo = Path(repo) if repo else None
    tok_path = Path(tok_path) if tok_path else repo / "naturalstories_RTS" / "all_stories.tok"
    wordinfo_path = Path(wordinfo_path) if wordinfo_path else repo / "naturalstories_RTS" / "processed_wordinfo.tsv"
    ud_path = Path(ud_path) if ud_path else repo / "parses" / "ud" / "stories-aligned.conllx"

    rt_index = load_rt_index(wordinfo_path)
    tokens_by_item = load_tokens(tok_path)
    ud_by_item = load_ud(ud_path)

    n_stories = 0
    with open(out, "w", encoding="utf-8") as fout:
        for item in sorted(tokens_by_item):
            toks = tokens_by_item[item]
            text = ""
            spr_records, zone_spans = [], {}
            for t in toks:
                if text:
                    text += " "
                cs = len(text)
                text += t["word"]
                ce = len(text)
                zone_spans[t["zone"]] = (cs, ce)
                stats = rt_index.get((item, t["zone"]), {})
                spr_records.append({
                    "zone": t["zone"], "word": t["word"], "char_start": cs, "char_end": ce,
                    "mean_rt": stats.get("mean_rt"), "gmean_rt": stats.get("gmean_rt"),
                    "sd_rt": stats.get("sd_rt"), "n_subj": stats.get("n_subj"),
                })
            ud_tokens, ud_sentences = align_ud_to_chars(text, zone_spans, ud_by_item.get(item, []))
            fout.write(json.dumps({
                "doc_id": f"naturalstories_{item}", "dataset": "naturalstories", "item": item,
                "text": text, "n_tokens": len(spr_records),
                "tokens": spr_records, "ud_tokens": ud_tokens, "ud_sentences": ud_sentences,
            }, ensure_ascii=False) + "\n")
            n_stories += 1
    print(f"[naturalstories] wrote {n_stories} stories "
          f"({sum(len(v) for v in tokens_by_item.values())} SPR tokens) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="path to cloned naturalstories repo (used to default the file paths)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", help="override path to all_stories.tok")
    ap.add_argument("--wordinfo", help="override path to processed_wordinfo.tsv")
    ap.add_argument("--ud-conll", dest="ud", help="override path to stories-aligned.conllx")
    args = ap.parse_args()
    if not args.repo and not (args.tok and args.wordinfo and args.ud):
        ap.error("provide --repo, or all of --tok/--wordinfo/--ud-conll")
    build(args.repo, Path(args.out), args.tok, args.wordinfo, args.ud)


if __name__ == "__main__":
    main()