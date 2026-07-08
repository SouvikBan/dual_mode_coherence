"""
annotate.py
===========
Augment the unified JSONL (produced by prepare_jsonl.py) with the three layers
needed for an entity- and grammatical-role-based information value function:

  1. Entities          - spaCy named-entity spans (typed).
  2. Coreference chains - clusters linking referring expressions (incl. pronouns)
                          ACROSS the context/target boundary, so you can tell which
                          target entities are GIVEN (already in context) vs NEW.
  3. Grammatical roles  - the silver dependency label of every token, and the
                          dependency label of each entity / coref-mention head
                          (i.e. the span root's predicted dep relation).

Design choices that matter for entity coherence
-----------------------------------------------
* Coreference is run on the CONCATENATION of context + target (+ optional
  post_context / alternatives). A coref model run on the target alone cannot
  link "he" in the target to "Thirlwall" in the context. We concatenate with
  known character offsets, run coref once, then map every mention back to the
  segment it falls in.
* The "entity mention" used for coherence is the coref mention (every referring
  expression, pronouns included), enriched with: its dependency head, its
  grammatical role (the silver dependency label of the span root), and the NER
  label if it overlaps a named entity. NER alone is centred on proper nouns and
  would drop pronouns and common-noun referents.
* For each item we also emit an `entity_grid` for the target: cluster_id -> the
  silver dependency label(s) realised by that entity in the target.

Coref backends (pluggable via --coref)
---------------------------------------
  maverick    (default) : pip install maverick-coref    (SOTA; weights from HF Hub)
  fastcoref             : pip install fastcoref          (LingMess; weights from HF Hub)
  coreferee             : pip install coreferee && python -m coreferee install en
                          (spaCy-native; lighter, lower accuracy, fully model-managed)
  none                  : skip coref (entities + dependency only)

spaCy model
-----------
  Prefers en_core_web_trf (transformer; most accurate parse + NER).
  Falls back to en_core_web_sm. Install one of:
     python -m spacy download en_core_web_trf
     python -m spacy download en_core_web_sm

Usage
-----
  python annotate.py --in clasp.jsonl        --out clasp.annotated.jsonl
  python annotate.py --in switchboard.jsonl  --out switchboard.annotated.jsonl --turn-sep "</s> <s>"
  # also annotate generated alternatives if your jsonl has an "alternatives": [..] field:
  python annotate.py --in with_alts.jsonl    --out with_alts.annotated.jsonl   --annotate-alternatives

Output adds, per item:
  "segments": { "<seg>": {"text","tokens":[...],"entities":[...]} , ... }
  "coref_clusters": [ {"cluster_id", "mentions":[{segment,char_start,char_end,text,
                        head_token,grammatical_role,ner_label}]} ]
  "entity_grid": { "<cluster_id>": ["<dep_label>", ...] }   # roles realised in the TARGET
  (and, if requested, the same structure under each alternative)
"""
import argparse
import json
from pathlib import Path

import spacy


# ----------------------------------------------------------------------------
# spaCy loading
# ----------------------------------------------------------------------------
def load_spacy(prefer_trf: bool = True):
    candidates = (["en_core_web_trf", "en_core_web_sm"]
                  if prefer_trf else ["en_core_web_sm", "en_core_web_trf"])
    for name in candidates:
        try:
            nlp = spacy.load(name)
            print(f"[spacy] loaded {name}")
            return nlp
        except Exception:
            continue
    raise SystemExit(
        "No spaCy English model found. Install one with:\n"
        "  python -m spacy download en_core_web_trf   (recommended)\n"
        "  python -m spacy download en_core_web_sm"
    )


# ----------------------------------------------------------------------------
# Coreference backends. Each returns clusters as lists of (start_char, end_char)
# spans over the *concatenated* document string.
# ----------------------------------------------------------------------------
class CorefBackend:
    def clusters(self, text: str):
        raise NotImplementedError


class MaverickBackend(CorefBackend):
    def __init__(self, device: str = "cpu", model: str = None, nlp=None):
        import torch
        from maverick import Maverick
        self.nlp = nlp  # used to tokenize input so we can use maverick's TOKEN offsets
        # PyTorch >=2.6 defaults torch.load(weights_only=True). Maverick's Lightning
        # checkpoint stores its config as an OmegaConf DictConfig, which is not an
        # allowed global under weights_only, so loading fails. The checkpoint is from a
        # trusted source (SapienzaNLP), so we load with weights_only=False just while
        # constructing the model, then restore torch.load.
        _orig_load = torch.load
        def _load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)
        torch.load = _load
        try:
            self.model = (Maverick(hf_name_or_path=model, device=device) if model
                          else Maverick(device=device))
        finally:
            torch.load = _orig_load
        # Singleton-capable checkpoints (litbank, preco) only EMIT singletons when
        # predict() is asked for them: maverick's `singletons` kwarg defaults to
        # False, which silently strips them regardless of checkpoint. This was the
        # cause of zero-singleton "litbank" annotations.
        self.want_singletons = bool(model) and any(
            k in model.lower() for k in ("litbank", "preco"))
        self._singletons_supported = None  # resolved on first predict
        if self.want_singletons:
            self._singleton_self_test()

    def _predict(self, sents):
        """predict() with singletons requested when the checkpoint supports them.
        Falls back (loudly) if the installed maverick predates the kwarg."""
        if self.want_singletons and self._singletons_supported is not False:
            try:
                out = self.model.predict(sents, singletons=True)
                self._singletons_supported = True
                return out
            except TypeError:
                self._singletons_supported = False
                print("[maverick] WARNING: installed maverick-coref does not accept "
                      "predict(..., singletons=True); singletons will be MISSING. "
                      "Upgrade the package (pip install -U maverick-coref).")
        return self.model.predict(sents)

    def _singleton_self_test(self):
        """Fail-fast probe: a litbank/preco checkpoint must produce a singleton on a
        canned example; if not, the checkpoint or package is not doing what the
        configuration claims, and every downstream given/new flag would be wrong."""
        probe = "John met Mary in Paris. He gave her a beautiful old book."
        try:
            cls = self.clusters(probe)
        except Exception as e:
            print(f"[maverick] WARNING: singleton self-test errored ({e}); "
                  f"proceeding without verification.")
            return
        n_sing = sum(1 for c in cls if len(c) == 1)
        print(f"[maverick] singleton self-test: {len(cls)} clusters, "
              f"{n_sing} singletons on probe sentence.")
        if n_sing == 0:
            print("[maverick] WARNING: ZERO singletons from a singleton-capable "
                  "checkpoint -- output will carry the OntoNotes signature. Check "
                  "the maverick-coref version and the checkpoint actually loaded "
                  "BEFORE running any annotation.")

    def clusters(self, text: str):
        # IMPORTANT: do NOT use maverick's clusters_char_offsets. Those are computed from
        # maverick's own internal sentence reconstruction and DRIFT (cumulatively, by 1+
        # chars) over long multi-sentence documents, corrupting mention spans. Instead we
        # pass our OWN spaCy tokenization (sentence_tokenized input) and map maverick's
        # token offsets back to exact character spans via our tokens.
        if self.nlp is None:
            raise RuntimeError("MaverickBackend needs a spaCy nlp for token alignment")
        doc = self.nlp(text)
        toks = list(doc)
        if not toks:
            return []
        try:
            sents = [[t.text for t in s] for s in doc.sents]
        except Exception:
            sents = [[t.text for t in toks]]
        if not sents:
            sents = [[t.text for t in toks]]
        out = self._predict(sents)  # token offsets index into the flattened token list
        clusters = []
        for cl in out["clusters_token_offsets"]:
            spans = []
            for (ts, te) in cl:  # INCLUSIVE token indices
                if 0 <= ts < len(toks) and 0 <= te < len(toks):
                    spans.append((toks[ts].idx, toks[te].idx + len(toks[te].text)))
            if spans:
                clusters.append(spans)
        return clusters


class FastCorefBackend(CorefBackend):
    def __init__(self, device: str = "cpu"):
        from fastcoref import FCoref
        self.model = FCoref(device=device)

    def clusters(self, text: str):
        pred = self.model.predict(texts=[text])[0]
        return [[(s, e) for (s, e) in cl] for cl in pred.get_clusters(as_strings=False)]


class CorefereeBackend(CorefBackend):
    """spaCy-native; resolves chains on a Doc. Returns char spans of each mention."""
    def __init__(self, nlp):
        import coreferee  # noqa: F401  (registers the pipe)
        if "coreferee" not in nlp.pipe_names:
            nlp.add_pipe("coreferee")
        self.nlp = nlp

    def clusters(self, text: str):
        doc = self.nlp(text)
        clusters = []
        for chain in doc._.coref_chains:
            cur = []
            for mention in chain:
                idxs = mention.token_indexes
                start_tok, end_tok = doc[min(idxs)], doc[max(idxs)]
                cur.append((start_tok.idx, end_tok.idx + len(end_tok.text)))
            clusters.append(cur)
        return clusters


def build_coref(name: str, nlp, device: str, model: str = None):
    if name == "none":
        return None
    if name == "maverick":
        return MaverickBackend(device=device, model=model, nlp=nlp)
    if name == "fastcoref":
        return FastCorefBackend(device=device)
    if name == "coreferee":
        return CorefereeBackend(nlp)
    raise ValueError(f"unknown coref backend: {name}")


# ----------------------------------------------------------------------------
# Per-segment annotation (tokens + NER), using a single spaCy Doc per segment.
# The grammatical role is the silver dependency label of the relevant token.
# ----------------------------------------------------------------------------
def annotate_segment(nlp, text: str):
    if not text:
        return {"text": "", "tokens": [], "entities": []}
    doc = nlp(text)
    tokens = [{
        "i": t.i, "text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_,
        "dep": t.dep_, "head": t.head.i, "ent_type": t.ent_type_, "ent_iob": t.ent_iob_,
        "char_start": t.idx, "char_end": t.idx + len(t.text),
    } for t in doc]
    entities = [{
        "text": ent.text, "label": ent.label_,
        "char_start": ent.start_char, "char_end": ent.end_char,
        "token_start": ent.start, "token_end": ent.end,
        "head_token": ent.root.i,
        "grammatical_role": ent.root.dep_,   # silver dependency label of the span root
    } for ent in doc.ents]
    return {"text": text, "tokens": tokens, "entities": entities, "_doc": doc}


def role_of_charspan(seg_ann, local_start, local_end):
    """Find the head (span root) of a mention inside a segment Doc and return its
    silver dependency label as the grammatical role, plus any overlapping NER label."""
    doc = seg_ann.get("_doc")
    if doc is None:
        return None
    span = doc.char_span(local_start, local_end, alignment_mode="expand")
    if span is None or len(span) == 0:
        return None
    root = span.root
    ner_label = None
    for ent in doc.ents:
        if ent.start_char < local_end and ent.end_char > local_start:
            ner_label = ent.label_
            break
    return {
        "head_token": root.i,
        "grammatical_role": root.dep_,   # silver dependency label
        "ner_label": ner_label,
    }


# ----------------------------------------------------------------------------
# Document-level annotation: per-segment parse + joint coref + entity grid.
# ----------------------------------------------------------------------------
def annotate_document(nlp, coref, segments: dict, join_sep: str = " "):
    """segments: ordered dict {seg_name: text}. Returns (segments_ann, coref_clusters, entity_grid)."""
    seg_ann = {name: annotate_segment(nlp, txt) for name, txt in segments.items()}

    offsets, parts, cursor = [], [], 0
    for name, txt in segments.items():
        if not txt:
            offsets.append((name, cursor, cursor))
            continue
        start = cursor
        parts.append(txt)
        cursor += len(txt)
        offsets.append((name, start, cursor))
        cursor += len(join_sep)
        parts.append(join_sep)
    joined = "".join(parts).rstrip()

    coref_clusters = []
    if coref is not None and joined.strip():
        try:
            raw_clusters = coref.clusters(joined)
        except Exception as e:
            print(f"[coref] WARNING failed on one document: {e}")
            raw_clusters = []
        for cid, cluster in enumerate(raw_clusters):
            mentions = []
            for (cs, ce) in cluster:
                seg_name, ls, le = _locate(cs, ce, offsets)
                if seg_name is None:
                    continue
                # clamp to the assigned segment so a mention that straddles the
                # context/target boundary stays within one segment, and make `text`
                # consistent with the (segment-local) char offsets.
                seg_text = seg_ann[seg_name]["text"]
                le = min(le, len(seg_text))
                info = role_of_charspan(seg_ann[seg_name], ls, le)
                m = {"segment": seg_name, "char_start": ls, "char_end": le,
                     "text": seg_text[ls:le]}
                if info:
                    m.update(info)
                mentions.append(m)
            if mentions:
                coref_clusters.append({"cluster_id": cid, "mentions": mentions})

    # entity grid for the TARGET: cluster_id -> silver dep label(s) realised in target
    entity_grid = {}
    for cl in coref_clusters:
        roles = [m.get("grammatical_role") for m in cl["mentions"]
                 if m["segment"] == "target" and m.get("grammatical_role")]
        if roles:
            entity_grid[str(cl["cluster_id"])] = roles

    for a in seg_ann.values():
        a.pop("_doc", None)
    return seg_ann, coref_clusters, entity_grid


def _locate(cs, ce, offsets):
    for name, s, e in offsets:
        if cs >= s and ce <= e:
            return name, cs - s, ce - s
    mid = (cs + ce) / 2
    for name, s, e in offsets:
        if s <= mid <= e:
            return name, max(0, cs - s), max(0, ce - s)
    return None, None, None


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--coref", default="maverick",
                    choices=["maverick", "fastcoref", "coreferee", "none"])
    ap.add_argument("--coref-model", default=None,
                    help="maverick checkpoint, e.g. sapienzanlp/maverick-mes-ontonotes "
                         "(default) or sapienzanlp/maverick-mes-litbank")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0 (maverick/fastcoref)")
    ap.add_argument("--spacy-sm", action="store_true",
                    help="prefer the small spaCy model over the transformer one")
    ap.add_argument("--annotate-alternatives", action="store_true",
                    help="also annotate each string in an 'alternatives' list (each vs context)")
    ap.add_argument("--turn-sep", default=None,
                    help="dialogue turn separator in 'context' (e.g. '</s> <s>'); replaced by a space for parsing")
    args = ap.parse_args()

    nlp = load_spacy(prefer_trf=not args.spacy_sm)
    coref = build_coref(args.coref, nlp, args.device, model=args.coref_model)

    n = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for item in iter_jsonl(args.inp):
            context = item.get("context", "") or ""
            target = item.get("target", "") or ""
            post = item.get("post_context") or ""
            if args.turn_sep:
                context = context.replace(args.turn_sep, " ")

            segments = {"context": context, "target": target}
            if post:
                segments["post_context"] = post

            seg_ann, clusters, grid = annotate_document(nlp, coref, segments)
            item["segments"] = seg_ann
            item["coref_clusters"] = clusters
            item["entity_grid"] = grid

            if args.annotate_alternatives and isinstance(item.get("alternatives"), list):
                alt_out = []
                for alt in item["alternatives"]:
                    a_seg, a_cl, a_grid = annotate_document(
                        nlp, coref, {"context": context, "target": str(alt)})
                    alt_out.append({"text": alt, "segments": a_seg,
                                    "coref_clusters": a_cl, "entity_grid": a_grid})
                item["alternatives_annotated"] = alt_out

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  ...{n} items")
    print(f"[done] annotated {n} items -> {args.out}")


if __name__ == "__main__":
    main()