"""
annotate_naturalstories.py
==========================
Annotate the story-level JSONL from prepare_naturalstories.py and emit one record
per SENTENCE with its summed reading time.

Sources of each layer
----------------------
  * Sentence segmentation  -> GOLD UD parse (embedded by prepare_naturalstories.py)
  * Grammatical role / dep  -> GOLD UD parse (the dep label of the span's head UD token)
  * Named entities          -> spaCy NER (run on the reconstructed text)
  * Coreference             -> story-level coref (maverick by default), resolved ONCE
                               over the whole story so long-range chains stay intact

The grammatical role of an entity / coref mention is the gold UD dependency label
of its head token (the token in the span whose head lies outside the span).

Per sentence the output exposes, alongside the reading times:
  entity_grid     {cluster_id: [gold dep label, ...]} realised in the sentence
  given_entities  clusters already mentioned in an EARLIER sentence (discourse-old)
  new_entities    clusters whose first mention is in THIS sentence (discourse-new)

Reading-time target (summed over the sentence's SPR tokens):
  reading_time_sum_mean, reading_time_sum_gmean, reading_time_mean_per_token
Optionally (--per-subject-rts processed_RTs.tsv) per-subject sentence sums.

Context defaults to the FULL preceding discourse of the story.

Usage
-----
  python prepare_naturalstories.py --repo naturalstories --out ns.jsonl
  python annotate_naturalstories.py --in ns.jsonl --out ns.sentences.annotated.jsonl
  python annotate_naturalstories.py --in ns.jsonl --out ns.annotated.jsonl \
      --per-subject-rts naturalstories/naturalstories_RTS/processed_RTs.tsv
"""
import argparse
import csv
import json
from collections import defaultdict

from annotate import load_spacy, build_coref


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_per_subject_rts(path):
    idx = defaultdict(lambda: defaultdict(dict))
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                item, zone, rt = int(r["item"]), int(r["zone"]), float(r["RT"])
            except (ValueError, KeyError):
                continue
            idx[item][zone][r["WorkerId"]] = rt
    return idx


def head_role_of_span(ud_tokens_by_g, gids_overlapping):
    """Given the UD tokens overlapping a character span, return the gold dep label
    of the span's syntactic head (the token whose head is outside the span)."""
    if not gids_overlapping:
        return None
    inside = set(gids_overlapping)
    roots = [g for g in gids_overlapping
             if ud_tokens_by_g[g]["head_g"] not in inside]  # head_g == -1 also qualifies
    pick = roots[0] if roots else gids_overlapping[0]
    return ud_tokens_by_g[pick]["dep"]


def overlapping_gids(ud_tokens, cs, ce):
    return [t["g"] for t in ud_tokens
            if t["char_start"] is not None and t["char_start"] < ce and t["char_end"] > cs]


def _build_windows(sent_spans, text, max_words, overlap_sents):
    """Group consecutive sentences into char windows under a word budget, with
    `overlap_sents` sentences shared between consecutive windows."""
    n = len(sent_spans)
    windows, i = [], 0
    while i < n:
        j, words = i, 0
        while j < n:
            w = max(1, len(text[sent_spans[j][0]:sent_spans[j][1]].split()))
            if j > i and words + w > max_words:
                break
            words += w
            j += 1
        windows.append((sent_spans[i][0], sent_spans[j - 1][1]))
        if j >= n:
            break
        i = max(i + 1, j - overlap_sents)
    return windows


def windowed_story_coref(coref, text, sent_spans, max_words, overlap_sents):
    """OPTIONAL. Resolve coref over overlapping windows of a story and stitch clusters
    that share an overlapping mention (IoU >= 0.5). Returns clusters as lists of global
    (char_start, char_end) spans. This is NOT needed for normal Natural Stories runs:
    maverick's DeBERTa encoder handles a full story in a single pass. It exists only
    for pathologically long documents, and trades long-range recall (chains can
    fragment across window seams) for shorter encoder inputs."""
    sent_spans = [s for s in sent_spans if s[0] is not None and s[1] is not None]
    if not sent_spans:
        return coref.clusters(text)
    windows = _build_windows(sent_spans, text, max_words, overlap_sents)
    if len(windows) <= 1:
        return coref.clusters(text)

    nodes, parent = [], []
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for wid, (cs, ce) in enumerate(windows):
        for cl in coref.clusters(text[cs:ce]):
            ids = []
            for (s, e) in cl:
                nid = len(nodes)
                nodes.append({"span": (s + cs, e + cs), "wid": wid})
                parent.append(nid)
                ids.append(nid)
            for k in range(1, len(ids)):
                union(ids[0], ids[k])

    def iou(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        uni = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / uni if uni > 0 else 0.0

    for x in range(len(nodes)):
        for y in range(x + 1, len(nodes)):
            if nodes[x]["wid"] != nodes[y]["wid"] and iou(nodes[x]["span"], nodes[y]["span"]) >= 0.5:
                union(x, y)

    groups = defaultdict(list)
    for nid in range(len(nodes)):
        groups[find(nid)].append(nodes[nid]["span"])
    return [sorted(set(g)) for g in groups.values()]


def annotate_story(nlp, coref, story, context_window=None, subj_rts=None,
                   coref_max_words=None, coref_overlap_sents=3):
    text = story["text"]
    item = story["item"]
    ud_tokens = story["ud_tokens"]
    ud_by_g = {t["g"]: t for t in ud_tokens}
    sents = story["ud_sentences"]

    # spaCy is used for NER only (sentence segmentation comes from gold UD)
    doc = nlp(text)
    ents = list(doc.ents)

    # zone -> sentence_index, for routing SPR reading times to sentences
    zone2sent = {}
    for s in sents:
        for z in s["zones"]:
            zone2sent[z] = s["sentence_index"]
    sent_spr = defaultdict(list)
    for tok in story["tokens"]:
        si = zone2sent.get(tok["zone"])
        if si is not None:
            sent_spr[si].append(tok)

    # ---- story-level coreference ----
    # maverick's DeBERTa encoder handles a full story in one pass (it does not have a
    # 512-token limit; DeBERTa uses relative-position attention), so the default is a
    # single whole-story pass. Optional windowing (coref_max_words) exists only for
    # pathologically long documents and trades long-range recall for shorter inputs.
    clusters = []
    if coref is not None and text.strip():
        try:
            if coref_max_words:
                sent_spans = [(s["char_start"], s["char_end"]) for s in sents]
                raw = windowed_story_coref(coref, text, sent_spans,
                                           coref_max_words, coref_overlap_sents)
            else:
                raw = coref.clusters(text)
        except Exception as e:
            print(f"[coref] WARNING failed on {story['doc_id']}: {e}")
            raw = []
        for cid, cl in enumerate(raw):
            mentions = []
            for (cs, ce) in cl:
                gids = overlapping_gids(ud_tokens, cs, ce)
                si = ud_by_g[gids[0]]["sentence_index"] if gids else None
                role = head_role_of_span(ud_by_g, gids)
                ner = next((e.label_ for e in ents if e.start_char < ce and e.end_char > cs), None)
                mentions.append({"sentence_index": si, "char_start": cs, "char_end": ce,
                                 "text": text[cs:ce], "grammatical_role": role, "ner_label": ner})
            clusters.append({"cluster_id": cid, "mentions": mentions})

    cluster_first_sent = {}
    for cl in clusters:
        sis = [m["sentence_index"] for m in cl["mentions"] if m["sentence_index"] is not None]
        if sis:
            cluster_first_sent[cl["cluster_id"]] = min(sis)

    records = []
    for s in sents:
        si = s["sentence_index"]
        cs, ce = s["char_start"], s["char_end"]
        if cs is None:
            continue
        gid2local = {g: k for k, g in enumerate(s["token_gids"])}

        # target tokens straight from the gold UD parse
        target_tokens = []
        for g in s["token_gids"]:
            t = ud_by_g[g]
            target_tokens.append({
                "i": gid2local[g], "form": t["form"], "pos": t["pos"], "dep": t["dep"],
                "head": gid2local.get(t["head_g"], -1),
                "char_start": (t["char_start"] - cs) if t["char_start"] is not None else None,
                "char_end": (t["char_end"] - cs) if t["char_end"] is not None else None,
                "zone": t["zone"],
            })

        # entities in this sentence; role = gold dep of entity head
        target_entities = []
        for e in ents:
            if e.start_char >= cs and e.end_char <= ce:
                gids = overlapping_gids(ud_tokens, e.start_char, e.end_char)
                target_entities.append({
                    "text": e.text, "label": e.label_,
                    "char_start": e.start_char - cs, "char_end": e.end_char - cs,
                    "grammatical_role": head_role_of_span(ud_by_g, gids),
                })

        # context = preceding discourse (full, or last K sentences)
        if context_window is None or si == 0:
            ctx_start = 0
        else:
            prev = sents[max(0, si - context_window)]
            ctx_start = prev["char_start"] if prev["char_start"] is not None else 0
        context_text = text[ctx_start:cs].strip()

        toks = sent_spr.get(si, [])
        mean_vals = [t["mean_rt"] for t in toks if t["mean_rt"] is not None]
        gmean_vals = [t["gmean_rt"] for t in toks if t["gmean_rt"] is not None]

        target_mentions, entity_grid = [], {}
        given, new = set(), set()
        for cl in clusters:
            cid = cl["cluster_id"]
            here = [m for m in cl["mentions"] if m["sentence_index"] == si]
            if not here:
                continue
            for m in here:
                ls = max(0, m["char_start"] - cs)
                le = max(ls, min(ce, m["char_end"]) - cs)
                target_mentions.append({
                    "cluster_id": cid, "char_start": ls, "char_end": le,
                    "text": text[cs:ce][ls:le],
                    "grammatical_role": m["grammatical_role"], "ner_label": m["ner_label"],
                })
            roles = [m["grammatical_role"] for m in here if m["grammatical_role"]]
            if roles:
                entity_grid[str(cid)] = roles
            (given if cluster_first_sent.get(cid, si) < si else new).add(cid)

        rec = {
            "doc_id": story["doc_id"], "dataset": "naturalstories", "item": item,
            "sentence_index": si, "context": context_text, "target": text[cs:ce],
            "n_spr_tokens": len(toks), "spr_zones": [t["zone"] for t in toks],
            "reading_time_sum_mean": sum(mean_vals) if mean_vals else None,
            "reading_time_sum_gmean": sum(gmean_vals) if gmean_vals else None,
            "reading_time_mean_per_token": (sum(mean_vals) / len(mean_vals)) if mean_vals else None,
            "segments": {
                "context": {"text": context_text},
                "target": {"text": text[cs:ce], "tokens": target_tokens, "entities": target_entities},
            },
            "target_mentions": target_mentions, "entity_grid": entity_grid,
            "given_entities": sorted(given), "new_entities": sorted(new),
        }

        if subj_rts is not None:
            zone_map = subj_rts.get(item, {})
            per_worker, per_worker_n = defaultdict(float), defaultdict(int)
            for t in toks:
                for worker, rt in zone_map.get(t["zone"], {}).items():
                    per_worker[worker] += rt
                    per_worker_n[worker] += 1
            rec["reading_times_by_subject"] = {
                w: {"rt_sum": round(v, 3), "n_tokens": per_worker_n[w]} for w, v in per_worker.items()
            }
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--coref", default="maverick", choices=["maverick", "fastcoref", "coreferee", "none"])
    ap.add_argument("--coref-model", default="sapienzanlp/maverick-mes-litbank",
                    help="maverick checkpoint. Default litbank (literary domain + predicts "
                         "singletons, suited to Natural Stories). Use sapienzanlp/maverick-mes-ontonotes "
                         "to match the dialogue/CLASP runs.")
    ap.add_argument("--coref-max-words", type=int, default=None,
                    help="OPTIONAL: split each story into overlapping windows of this many words "
                         "for coref. OFF by default — maverick's DeBERTa encoder handles a full "
                         "story in one pass; only set this for pathologically long documents.")
    ap.add_argument("--coref-overlap-sents", type=int, default=3,
                    help="sentences shared between consecutive windows (only used if --coref-max-words is set)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--spacy-sm", action="store_true")
    ap.add_argument("--context-window", type=int, default=None,
                    help="limit context to the last K sentences (default: full preceding discourse)")
    ap.add_argument("--per-subject-rts", default=None,
                    help="path to processed_RTs.tsv to also emit per-subject sentence RT sums")
    args = ap.parse_args()

    nlp = load_spacy(prefer_trf=not args.spacy_sm)
    coref = build_coref(args.coref, nlp, args.device, model=args.coref_model)
    subj_rts = load_per_subject_rts(args.per_subject_rts) if args.per_subject_rts else None

    n_sent = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for story in iter_jsonl(args.inp):
            for rec in annotate_story(nlp, coref, story,
                                      context_window=args.context_window, subj_rts=subj_rts,
                                      coref_max_words=args.coref_max_words,
                                      coref_overlap_sents=args.coref_overlap_sents):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_sent += 1
            print(f"  {story['doc_id']}: done")
    print(f"[done] wrote {n_sent} sentence records -> {args.out}")


if __name__ == "__main__":
    main()