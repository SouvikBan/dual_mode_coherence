#!/usr/bin/env python3
"""Shared code for turning raw generations into N annotated alternatives.

Used by annotate_clasp.py and annotate_ns.py.

Selection is the one in compute_information_value.py (dmg-illc/information-
value) with --separator spacy: each raw is stripped and, if it is not empty,
replaced by the first sentence that spaCy (en_core_web_sm, full pipeline)
finds in it. Nothing else is filtered.

Texts that contain an HTML tag or a letter outside a-z/A-Z are held back
and used (in raw order) only if the other raws do not give N.

The raws are then taken in their original order, parsed with Stanza (UD)
and annotated with seeded CorPipe against the frozen gold context. A raw is
skipped only when this does not work: empty text, Stanza error, or CorPipe
error. The first N that work are the alternatives. Duplicates are kept; an
identical text is annotated once.

The alternative text is exactly the reference text. Stanza parses it as one
sentence: line breaks are given to Stanza as spaces (same length, so
character offsets still match the text).
"""

from __future__ import annotations

import copy
import csv
import argparse
import hashlib
import json
import os
import re
import socket
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Sequence

STRATEGIES = (
    "ancestral", "temp_075", "temp_125", "nucleus_08", "nucleus_085",
    "nucleus_09", "nucleus_095", "typical_02", "typical_03", "typical_085", "typical_095",
)
DEFAULT_MODEL = "ufal/corpipe26-twostage-corefud1.4-large-260702"
SAMPLE_ERRORS = (ValueError, IndexError, AssertionError, KeyError)

# Same label inventory as the information-value script.
UD_RELATIONS = set(
    "acl advcl advmod amod appos aux case cc ccomp clf compound conj cop csubj dep det "
    "discourse dislocated expl fixed flat goeswith iobj list mark nmod nsubj nummod obj obl "
    "orphan parataxis punct reparandum root vocative xcomp".split()
)
IV_ALIASES = {"nsubjpass": "nsubj", "csubjpass": "csubj", "dobj": "obj", "auxpass": "aux"}


# ---------------------------------------------------------------- small io

def normalise_text(text: str) -> str:
    return " ".join((text or "").split())


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic_json(path, value):
    """Write to a process-unique temporary file, then rename (safe when
    several workers write into the same folder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_token_csv(path, escaped: bool):
    """escaped=True for files written with QUOTE_NONE and escapechar='\\'
    (the CLASP token files); False for the Natural Stories CSVs."""
    kwargs = {"delimiter": ";"}
    if escaped:
        kwargs.update(quoting=csv.QUOTE_NONE, escapechar="\\")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, **kwargs))


# ------------------------------------------------------------ gold entities

OPEN_RE = re.compile(r"\((\d+)-")
CLOSE_RE = re.compile(r"(\d+)\)")


def entity_spans(layers: Sequence[str]) -> list[tuple[int, int, int]]:
    """(start, end, cluster) token spans from a GUM-style entity column."""
    stack: list[tuple[int, int]] = []
    spans: list[tuple[int, int, int]] = []

    def close(index, expected=None):
        if not stack:
            raise ValueError(f"closing unopened entity at token {index}")
        if expected is None:
            entity, start = stack.pop()
        else:
            hits = [i for i, (entity, _) in enumerate(stack) if entity == expected]
            if not hits:
                raise ValueError(f"closed unopened entity {expected} at token {index}")
            entity, start = stack.pop(hits[-1])
        spans.append((start, index, entity))

    for index, layer in enumerate(layers):
        if not layer or layer == "_":
            continue
        layer = layer.removeprefix("Entity=")
        position = 0
        while position < len(layer):
            opened = OPEN_RE.match(layer, position)
            if opened:
                stack.append((int(opened.group(1)), index))
                position = opened.end()
                continue
            closed = CLOSE_RE.match(layer, position) if position == 0 or layer[position - 1] == ")" else None
            if closed:
                close(index, int(closed.group(1)))
                position = closed.end()
                continue
            if layer[position] == ")":
                close(index)
            position += 1
    if stack:
        raise ValueError(f"unclosed entities: {[entity for entity, _ in stack]}")
    return sorted(spans, key=lambda span: (span[0], -span[1], span[2]))


def drop_space_tokens(rows, spans):
    """Remove whitespace-only tokens (spaCy SPACE tokens such as U+00A0 in the
    CLASP files) and move entity boundaries to the nearest kept token."""
    keep = [i for i, row in enumerate(rows) if row["form"].strip()]
    new_index = {old: new for new, old in enumerate(keep)}
    kept_spans = []
    for start, end, cluster in spans:
        inside = [new_index[i] for i in range(start, end + 1) if i in new_index]
        if inside:
            kept_spans.append((inside[0], inside[-1], cluster))
    return [rows[i] for i in keep], kept_spans


def align_offsets(text: str, forms: Sequence[str]):
    """Character offsets of consecutive forms in text, or None."""
    offsets, position = [], 0
    for form in forms:
        index = text.find(form, position)
        if index < 0 or text[position:index].strip():
            return None
        offsets.append((index, index + len(form)))
        position = index + len(form)
    return None if text[position:].strip() else offsets


# -------------------------------------------------------- sentence helpers

def validate_sentence(sentence):
    """Reject incomplete syntax, malformed trees and invalid entity spans."""
    tokens = sentence["tokens"]
    if not tokens or [int(t["id"]) for t in tokens] != list(range(1, len(tokens) + 1)):
        raise ValueError("empty sentence or non-contiguous word IDs")
    heads = {int(t["id"]): int(t["head"]) for t in tokens}
    if sum(head == 0 for head in heads.values()) != 1:
        raise ValueError("dependency tree must have one root")
    for token in tokens:
        if not token.get("text") or token.get("deprel") in (None, "", "_"):
            raise ValueError("missing token text or dependency label")
        if not token.get("upos") or token["upos"] == "_":
            raise ValueError("missing POS annotation")
        start, end = int(token["char_start"]), int(token["char_end"])
        if not 0 <= start < end <= len(sentence["text"]):
            raise ValueError("invalid character span")
        seen, current = set(), int(token["id"])
        while current:
            if current in seen or current not in heads:
                raise ValueError("dependency cycle or out-of-range head")
            seen.add(current)
            current = heads[current]
    for mention in sentence.get("mentions", []):
        if not 0 <= int(mention["token_start"]) <= int(mention["token_end"]) < len(tokens):
            raise ValueError("invalid entity span")
        int(mention["cluster_id"])


def check_iv_labels(sentence, where: str):
    """Fail here instead of in the IV script when a label is not UD."""
    bad = set()
    for token in sentence["tokens"]:
        base = str(token["deprel"]).casefold().split(":", 1)[0]
        base = IV_ALIASES.get(base, base)
        if base not in UD_RELATIONS:
            bad.add(token["deprel"])
    if bad:
        raise ValueError(f"{where}: dependency labels the IV script rejects: {sorted(bad)}")


def clean_sentence(sentence):
    sentence = copy.deepcopy(sentence)
    if not isinstance(sentence.get("mentions"), list):
        raise ValueError("missing entity annotation; use an empty mentions list for zero entities")
    validate_sentence(sentence)
    events = [[] for _ in sentence["tokens"]]
    for mention in sentence["mentions"]:
        start, end = int(mention["token_start"]), int(mention["token_end"])
        entity = int(mention["cluster_id"])
        if start == end:
            events[start].append((1, entity, f"({entity})"))
        else:
            events[start].append((0, -end, f"({entity}"))
            events[end].append((2, -start, f"{entity})"))
    for token, values in zip(sentence["tokens"], events):
        token["form"] = token["text"]
        token["pos"] = token.get("xpos") or token["upos"]
        token["entity_layer"] = "|".join(value for _, _, value in sorted(values)) or "_"
    return sentence


def clean_branch(branch):
    result = dict(branch)
    for key in ("context_cluster_ids", "n_given_mentions", "n_new_mentions", "grammatical_sentences"):
        result.pop(key, None)
    if not result.get("sentences"):
        raise ValueError("annotation returned no sentences")
    result["sentences"] = [clean_sentence(s) for s in result["sentences"]]
    result["annotation_source"] = "stanza_ud_seeded_corpipe_entities"
    return result


def raw_mention(sentence, token_start, token_end, cluster_id):
    tokens = sentence["tokens"]
    char_start = int(tokens[token_start]["char_start"])
    char_end = int(tokens[token_end]["char_end"])
    return {"token_start": int(token_start), "token_end": int(token_end),
            "char_start": char_start, "char_end": char_end,
            "text": sentence["text"][char_start:char_end], "cluster_id": int(cluster_id)}


# ------------------------------------------------ UD v1 -> v2 (NS gold)

V1_TO_V2 = {"dobj": "obj", "nsubjpass": "nsubj:pass", "csubjpass": "csubj:pass",
            "auxpass": "aux:pass", "mwe": "fixed", "name": "flat",
            "foreign": "flat:foreign", "remnant": "orphan"}


def convert_ud1_to_ud2(tokens):
    """Relabel a UD v1 tree (Natural Stories CSV) with UD v2 labels.

    Rules as in udapi ud.Convert1to2: renamed relations, neg -> advmod
    (det for determiners), and nmod(:tmod/:npmod) -> obl(:...) when the
    governor is a verb, adjective, adverb or a copular predicate.
    """
    by_id = {int(t["id"]): t for t in tokens}
    has_cop = {int(t["head"]) for t in tokens if str(t["deprel"]).split(":")[0] == "cop"}

    def predicate(token):
        xpos = str(token.get("xpos") or token.get("upos") or "")
        return xpos.startswith(("VB", "MD", "JJ", "RB")) or int(token["id"]) in has_cop

    for token in tokens:
        label = str(token["deprel"])
        base, _, sub = label.partition(":")
        new = label
        if base in V1_TO_V2 and not sub:
            new = V1_TO_V2[base]
        elif base == "neg":
            new = "det" if str(token.get("xpos")) in ("DT", "PDT") else "advmod"
        elif base == "nmod" and sub in ("", "tmod", "npmod"):
            parent = by_id.get(int(token["head"]))
            if parent is not None and predicate(parent):
                new = "obl" + (f":{sub}" if sub else "")
        if new != label:
            token["source_deprel"] = label
            token["deprel"] = new
    return tokens


# ------------------------------------------------------------ Stanza UD

class SeveralSentences(ValueError):
    pass


class StanzaUD:
    """English Stanza with the same models for free text (alternatives) and
    for gold tokenisation (CLASP context/targets, optionally NS)."""

    def __init__(self, model_dir=None, use_gpu=False, package="default", batch_size=64):
        self.kwargs = {"lang": "en", "use_gpu": use_gpu, "verbose": False,
                       "pos_batch_size": 3000, "depparse_batch_size": 3000}
        if package and package != "default":
            self.kwargs["package"] = package
        if model_dir:
            self.kwargs["dir"] = str(model_dir)
        self.batch_size = batch_size
        self.raw = self._pipeline(tokenize_no_ssplit=True)
        self._pretokenized = None

    def _pipeline(self, **extra):
        import stanza
        if hasattr(stanza, "DownloadMethod"):  # use local models, download only if missing
            extra.setdefault("download_method", stanza.DownloadMethod.REUSE_RESOURCES)
        error = None
        for processors in ("tokenize,mwt,pos,lemma,depparse", "tokenize,pos,lemma,depparse"):
            try:
                return stanza.Pipeline(processors=processors, **self.kwargs, **extra)
            except Exception as caught:  # mwt is not in every English package
                error = caught
        raise error

    @property
    def pretokenized(self):
        if self._pretokenized is None:
            self._pretokenized = self._pipeline(tokenize_pretokenized=True)
        return self._pretokenized

    @staticmethod
    def _word(word):
        return {"id": int(word.id), "text": word.text, "lemma": word.lemma,
                "upos": word.upos or "X", "xpos": word.xpos or "_", "feats": word.feats or "_",
                "head": int(word.head) if word.head is not None else -1,
                "deprel": word.deprel or "_"}

    def _sentence(self, document, text):
        if len(document.sentences) != 1:
            raise SeveralSentences(f"Stanza found {len(document.sentences)} sentences")
        tokens = []
        for token in document.sentences[0].tokens:
            for word in token.words:
                start = getattr(word, "start_char", None)
                end = getattr(word, "end_char", None)
                if start is None or end is None or len(token.words) > 1:
                    start, end = token.start_char, token.end_char
                item = self._word(word)
                if not item["text"] or item["text"] == "<UNK>":
                    item["text"] = text[start:end]
                item["lemma"] = item["lemma"] or item["text"]
                item.update(deps="_", misc="_", char_start=int(start), char_end=int(end),
                            grammar_source="stanza_ud")
                tokens.append(item)
        sentence = {"text": text, "tokens": tokens, "grammar_source": "stanza_ud"}
        validate_sentence({**sentence, "mentions": []})
        return sentence

    @staticmethod
    def one_line(text):
        """Line breaks -> spaces, same length. The reference keeps line breaks
        inside an alternative; Stanza would treat a blank line as a sentence
        break. Character offsets stay valid for the original text."""
        return "".join(" " if ch.isspace() else ch for ch in text)

    def parse_many(self, texts: Sequence[str]) -> list:
        """One sentence per text; an Exception object marks a failure."""
        import stanza
        out = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            try:
                documents = [stanza.Document([], text=self.one_line(text)) for text in chunk]
                if hasattr(self.raw, "bulk_process"):
                    documents = self.raw.bulk_process(documents)
                else:
                    documents = [self.raw(document) for document in documents]
                parsed = []
                for document, text in zip(documents, chunk):
                    try:
                        parsed.append(self._sentence(document, text))
                    except SAMPLE_ERRORS as error:
                        parsed.append(error)
                out.extend(parsed)
            except Exception:
                for text in chunk:
                    try:
                        out.append(self._sentence(self.raw(self.one_line(text)), text))
                    except Exception as error:  # isolate one bad text
                        out.append(ValueError(f"stanza: {error}"))
        return out

    def parse_document(self, text: str) -> list:
        """Every sentence of a multi-sentence text, in the project's sentence
        format. parse_many insists on one sentence per text because an
        alternative must be a single sentence; a CLASP pre-context is a whole
        passage, so it needs its own entry point. Character offsets are
        relative to `text`."""
        flat = self.one_line(text)
        if not flat.strip():
            return []
        document = self.raw(flat)
        out = []
        for index, parsed in enumerate(document.sentences):
            tokens = []
            for token in parsed.tokens:
                for word in token.words:
                    start = getattr(word, "start_char", None)
                    end = getattr(word, "end_char", None)
                    if start is None or end is None or len(token.words) > 1:
                        start, end = token.start_char, token.end_char
                    item = self._word(word)
                    if not item["text"] or item["text"] == "<UNK>":
                        item["text"] = flat[start:end]
                    item["lemma"] = item["lemma"] or item["text"]
                    item.update(deps="_", misc="_", char_start=int(start), char_end=int(end),
                                grammar_source="stanza_ud")
                    tokens.append(item)
            if not tokens:
                continue
            span = flat[tokens[0]["char_start"]:tokens[-1]["char_end"]]
            sentence = {"sentence_index": index, "text": span, "tokens": tokens,
                        "mentions": [], "grammar_source": "stanza_ud"}
            validate_sentence(sentence)
            out.append(sentence)
        return out

    def parse_pretokenized(self, sentences: Sequence[Sequence[str]]):
        """Per sentence, per gold token: the list of Stanza words (MWT-safe)."""
        if not sentences:
            return []
        document = self.pretokenized([list(forms) for forms in sentences])
        if len(document.sentences) != len(sentences):
            raise ValueError("Stanza changed the gold sentence segmentation")
        result = []
        for forms, sentence in zip(sentences, document.sentences):
            if len(sentence.tokens) != len(forms):
                raise ValueError("Stanza changed the gold tokenisation")
            result.append([[self._word(word) for word in token.words] for token in sentence.tokens])
        return result


def build_gold_sentence(forms, text, offsets, spans, words_by_token=None, gold_syntax=None,
                        meta=None, syntax_label="stanza_ud_on_gold_tokens", token_ids=None):
    """Sentence dict with gold tokens and entity spans.

    words_by_token: Stanza words per gold token (pretokenised parse), or
    gold_syntax: [(head, deprel, pos)] per gold token (NS CSV).
    """
    if offsets is None:
        text = " ".join(forms)
        offsets, cursor = [], 0
        for form in forms:
            offsets.append((cursor, cursor + len(form)))
            cursor += len(form) + 1
    tokens, first_word, last_word = [], {}, {}
    for index, form in enumerate(forms):
        start, end = offsets[index]
        if gold_syntax is not None:
            head, deprel, pos = gold_syntax[index]
            words = [{"id": index + 1, "text": form, "lemma": "_", "upos": pos, "xpos": pos,
                      "feats": "_", "head": int(head), "deprel": deprel}]
        else:
            words = words_by_token[index]
        first_word[index] = len(tokens)
        for word in words:
            if int(word["id"]) != len(tokens) + 1:
                raise ValueError("word ids are not contiguous")
            item = dict(word)
            item.update(source_token_id=token_ids[index] if token_ids else str(index + 1), deps="_", misc="_",
                        char_start=start, char_end=end, grammar_source=syntax_label)
            if not item.get("text") or item["text"] == "<UNK>":
                item["text"] = form
            if not item.get("lemma") or item["lemma"] in ("<UNK>", None):
                item["lemma"] = item["text"]
            tokens.append(item)
        last_word[index] = len(tokens) - 1
    sentence = {**(meta or {}), "text": text, "tokens": tokens,
                "grammar_source": syntax_label, "entity_source": "manual_gum_csv"}
    sentence["mentions"] = [raw_mention(sentence, first_word[start], last_word[end], cluster)
                            for start, end, cluster in spans]
    validate_sentence(sentence)
    return clean_sentence(sentence)


# ---------------------------------------------------- annotation of pools

class AnnotationCache:
    """text -> ('ok', branch) or (failure_phase, message), per context."""

    def __init__(self, enabled=True):
        self.enabled, self.store = enabled, {}

    def clear(self):
        self.store.clear()


def reference_sentences(raws, nlp):
    """compute_information_value.py, --separator spacy:
    alternative = raw.strip(); if non-empty: first spaCy sentence."""
    texts = [raw.strip() for raw in raws]
    positions = [i for i, text in enumerate(texts) if text != ""]
    for i, doc in zip(positions, nlp.pipe([texts[i] for i in positions], batch_size=64)):
        texts[i] = next(doc.sents).text
    return texts


def annotate_texts(texts, context, annotator, parser, max_tokens, cache, job_id):
    """Stanza + CorPipe for each text; None where it does not work."""
    store = cache.store if cache.enabled else {}
    todo = [t for t in dict.fromkeys(texts) if t not in store]
    for text in todo:
        if text == "":
            store[text] = ("empty", "")
    todo = [t for t in todo if t not in store]
    if todo:
        branches = []
        for text, sentence in zip(todo, parser.parse_many(todo)):
            if isinstance(sentence, SeveralSentences):
                store[text] = ("stanza_several_sentences", str(sentence))
            elif isinstance(sentence, Exception):
                store[text] = ("stanza_failed", str(sentence))
            elif max_tokens and len(sentence["tokens"]) > max_tokens:
                store[text] = ("too_many_tokens", f"{len(sentence['tokens'])} Stanza words")
            else:
                branches.append({"id": f"u{len(branches)}", "kind": "generated_alternative",
                                 "text": text, "grammatical_sentences": [sentence]})
        if branches:
            outputs = run_corpipe(annotator, context, branches, f"{job_id}-{len(store)}")
            for branch, output in zip(branches, outputs):
                text = branch["text"]
                if isinstance(output, Exception):
                    store[text] = ("corpipe_failed", str(output))
                    continue
                try:
                    result = clean_branch(output)
                    if len(result["sentences"]) != 1:
                        raise ValueError("alternative must have one annotated sentence")
                    if len(result["sentences"][0]["tokens"]) != len(branch["grammatical_sentences"][0]["tokens"]):
                        raise ValueError("CorPipe changed the tokenisation")
                    store[text] = ("ok", result)
                except SAMPLE_ERRORS as error:
                    store[text] = ("invalid_annotation", str(error))
    return [store[text] for text in texts]


def run_corpipe(annotator, context, branches, job_id):
    """Batch call; on a sample error, isolate the failing branch."""
    try:
        outputs = annotator.annotate_branches(context, branches, job_id)
        if len(outputs) != len(branches):
            raise RuntimeError("CorPipe returned the wrong number of branches")
    except SAMPLE_ERRORS:
        outputs = []
        for branch in branches:
            try:
                result = annotator.annotate_branches(context, [branch], f"{job_id}-{branch['id']}")
                if len(result) != 1:
                    raise RuntimeError("CorPipe returned the wrong number of branches")
                outputs.extend(result)
            except SAMPLE_ERRORS as error:
                outputs.append(error)
    for branch, output in zip(branches, outputs):
        if not isinstance(output, Exception) and output.get("id") != branch["id"]:
            raise RuntimeError("CorPipe changed branch order/identity")
    return outputs


# An HTML/XML tag, including model tokens written like tags (<unused60>).
HTML_TAG_RE = re.compile(r"<!--|<![A-Za-z]|</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>")


def deprioritised(text):
    """Reasons to use this alternative only if needed: an HTML tag, or a
    letter outside a-z/A-Z (non-English script, accented letter)."""
    reasons = []
    if HTML_TAG_RE.search(text):
        reasons.append("html")
    if any(ch.isalpha() and not ch.isascii() for ch in text):
        reasons.append("non_english")
    return reasons


# A complete sentence ends in final punctuation once trailing quotes/brackets
# are removed. The target is always one complete sentence; without this test an
# alternative can be the 120-token cut-off of an unfinished one.
SENTENCE_CLOSERS = '"\u201d\u2019\')]}\u00bb'
SENTENCE_FINAL = tuple('.!?\u2026\u3002\uff01\uff1f')


def complete_sentence(text: str) -> bool:
    return text.rstrip().rstrip(SENTENCE_CLOSERS).rstrip().endswith(SENTENCE_FINAL)


def annotate_pool(raws, strategy, context, annotator, parser, nlp, cache, args, job_id):
    """Raws in their original order until args.n are annotated.

    --require-complete-sentence drops a text whose first sentence does not end
    in final punctuation (the reference keeps it; the target never is one).
    --no-hold-back uses HTML / non-English texts in raw order like any other
    sample instead of deferring them to the back of the queue.
    --lowercase-alternatives folds the text to lower case, for CLASP, whose
    gold context and targets are lower case.

    Raws are taken in four tiers, each only when the one before it cannot
    reach N, so the sample stays as close to the model's own distribution as
    the filters allow while still being filled:

      1. complete first sentence, no HTML tag and no non-English letter;
      2. incomplete first sentence (the 120-token generation limit cut it off);
      3. held-back HTML / non-English text whose first sentence is complete;
      4. held-back HTML / non-English text whose first sentence is incomplete.

    N is then reached unconditionally. If all four tiers together still fall
    short, the kept alternatives are cycled through and repeated until N is
    met, because a short pool makes the whole document incomplete and the IV
    script refuses the entire file, which loses far more than a repeated draw.
    Cycling repeats every kept alternative about equally, so the empirical
    distribution of what was actually sampled is preserved; it adds no
    information, and the copies are marked "padding_duplicate" so they can be
    counted or dropped. --no-pad-to-n fails loudly instead.

    Every alternative records the tier it came from in "deprioritised"
    ("incomplete", "html", "non_english", "padding_duplicate"), so any draw
    that is not a tier-1 draw can be identified downstream.
    """
    if not all(isinstance(raw, str) for raw in raws):
        raise ValueError("raws must be a list of strings")
    require_complete = getattr(args, "require_complete_sentence", False)
    hold_back = not getattr(args, "no_hold_back", False)
    lowercase = getattr(args, "lowercase_alternatives", False)
    incomplete_fallback = getattr(args, "incomplete_fallback", True)
    pad_to_n = getattr(args, "pad_to_n", True)
    # tiers 2..4, filled while tier 1 streams
    tier2, tier3, tier4 = [], [], []
    kept, position = [], 0
    discarded, examples, held_back = Counter(), [], Counter()

    def annotate(batch):  # batch: [(raw_index, text, reasons)]
        results = annotate_texts([text for _, text, _ in batch], context, annotator, parser,
                                 args.max_tokens, cache, f"{job_id}-{strategy}")
        for (raw_index, text, reasons), (status, value) in zip(batch, results):
            if status != "ok":
                discarded[status] += 1
                if len(examples) < 5:
                    examples.append({"raw_index": raw_index, "reason": status,
                                     "text": text[:120], "error": value[:300]})
                continue
            alternative = copy.deepcopy(value)
            alternative.update({"id": f"{strategy}_{raw_index}", "kind": "generated_alternative",
                                "strategy": strategy, "raw_index": raw_index,
                                "sample_index": len(kept) + 1, "text": text,
                                "deprioritised": reasons})
            kept.append(alternative)

    def drain(bucket):
        used = 0
        while len(kept) < args.n and used < len(bucket):
            batch = bucket[used:used + min(args.batch_size, args.n - len(kept))]
            used += len(batch)
            annotate(batch)
        return used

    # tier 1 streams: scan the raws in order, annotate the clean complete ones
    # as they appear and set the rest aside for their tier.
    while len(kept) < args.n and position < len(raws):
        start = position
        chunk = raws[start:start + min(args.batch_size, args.n - len(kept))]
        position += len(chunk)
        tier1 = []
        for offset, text in enumerate(reference_sentences(chunk, nlp)):
            if lowercase:
                text = text.lower()
            raw_index = start + offset + 1  # 1-based
            flags = deprioritised(text)
            for reason in flags:
                held_back[reason] += 1
            whole = bool(text) and (not require_complete or complete_sentence(text))
            if not whole and not incomplete_fallback:
                discarded["incomplete_sentence"] += 1
                if len(examples) < 5:
                    examples.append({"raw_index": raw_index, "reason": "incomplete_sentence",
                                     "text": text[:120], "error": ""})
                continue
            reasons = list(flags) + ([] if whole else ["incomplete"])
            item = (raw_index, text, reasons)
            flagged = bool(flags) and hold_back
            if flagged:
                (tier3 if whole else tier4).append(item)
            elif whole:
                tier1.append(item)
            else:
                tier2.append(item)
        if tier1:
            annotate(tier1)

    used = {"incomplete": drain(tier2), "held_back_complete": drain(tier3),
            "held_back_incomplete": drain(tier4)}

    # N is reached unconditionally: repeat kept draws in a cycle if need be.
    n_padded = 0
    if len(kept) < args.n and pad_to_n:
        if not kept:
            raise ValueError(f"{job_id}/{strategy}: no raw could be annotated at all "
                             f"({dict(discarded)}); nothing to pad from")
        print(f"WARNING {job_id}/{strategy}: only {len(kept)} of {args.n} distinct "
              f"alternatives; padding by repeating kept draws", flush=True)
        source = list(kept)
        while len(kept) < args.n:
            original = source[n_padded % len(source)]
            copy_of = copy.deepcopy(original)
            copy_of["id"] = f"{original['id']}_pad{n_padded + 1}"
            copy_of["sample_index"] = len(kept) + 1
            copy_of["deprioritised"] = list(original["deprioritised"]) + ["padding_duplicate"]
            copy_of["padded_from"] = original["id"]
            kept.append(copy_of)
            n_padded += 1

    flags_of = lambda a: set(a["deprioritised"])
    qa = {"n_raws": len(raws), "n_tried": position, "n_annotated": len(kept),
          "n_distinct": len(kept) - n_padded,
          "tiers": {"complete": sum(1 for a in kept if not flags_of(a)),
                    "incomplete": sum(1 for a in kept
                                      if "incomplete" in flags_of(a)
                                      and not flags_of(a) & {"html", "non_english"}),
                    "held_back": sum(1 for a in kept if flags_of(a) & {"html", "non_english"}),
                    "padding_duplicate": n_padded},
          "available": {"incomplete": len(tier2), "held_back_complete": len(tier3),
                        "held_back_incomplete": len(tier4)},
          "used": used,
          "held_back": {"texts": len(tier3) + len(tier4), **held_back},
          "held_back_used": sum(1 for a in kept if flags_of(a) & {"html", "non_english"}),
          "incomplete": {"available": len(tier2), "used": used["incomplete"]},
          "discarded": dict(discarded), "discard_examples": examples}
    return kept, qa


def corpipe_context(sentences, annotator, job_id):
    """Replace the manual entity annotation of a context with CorPipe's own.

    Tokens and syntax are left exactly as they are, so a run with this and a
    run without it differ only in where the context clusters come from: the
    manual annotation, or an ordinary CorPipe pass over the same tokens. The
    alternatives are then seeded from whichever context is frozen here.
    """
    if not sentences:
        return []
    predicted = annotator.annotate_document(copy.deepcopy(sentences), f"{job_id}-context")
    if len(predicted) != len(sentences):
        raise ValueError(f"{job_id}: CorPipe returned {len(predicted)} context sentences, "
                         f"expected {len(sentences)}")
    output = []
    for gold, silver in zip(sentences, predicted):
        merged = copy.deepcopy(gold)
        merged["mentions"] = copy.deepcopy(silver.get("mentions", []))
        merged["entity_source"] = "corpipe_on_gold_tokens"
        if [t["text"] for t in merged["tokens"]] != [t["text"] for t in silver["tokens"]]:
            raise ValueError(f"{job_id}: CorPipe changed the context tokenisation")
        output.append(clean_sentence(merged))
    return output


def annotate_target(text, context, annotator, parser, nlp, args, job_id, cache=None):
    """Annotate an observed target exactly as an alternative is annotated:
    Stanza on the plain text, then seeded CorPipe against the frozen gold
    context. Used by --target-annotation corpipe so that the target and the
    alternatives carry the same kind of tokens, syntax and entity annotation."""
    text = text.strip()
    if getattr(args, "lowercase_alternatives", False):
        text = text.lower()
    status, value = annotate_texts([text], context, annotator, parser, args.max_tokens,
                                   cache or AnnotationCache(False), f"{job_id}-target")[0]
    if status != "ok":
        raise ValueError(f"{job_id}: the target could not be annotated ({status}: {value})")
    branch = copy.deepcopy(value)
    branch.update({"id": "target", "kind": "observed_target", "text": text,
                   "annotation_source": "stanza_ud_seeded_corpipe_entities"})
    return branch


def summary_line(job_id, strategy, qa, n):
    discarded = ", ".join(f"{k}={v}" for k, v in qa["discarded"].items()) or "none"
    flag = "" if qa["n_annotated"] >= n else f"  SHORT {qa['n_annotated']}/{n}"
    deferred = qa["held_back"]["texts"]
    flagged = (f"{qa['held_back_used']} of {deferred} held-back used" if deferred
               else f"{qa['held_back_used']} html/non-english kept in order")
    tiers = qa.get("tiers", {})
    extra = [f"{k}={v}" for k, v in tiers.items() if k != "complete" and v]
    if extra:
        flagged += ", tiers " + " ".join(extra)
    return (f"{job_id}/{strategy}: tried {qa['n_tried']}/{qa['n_raws']} raws, kept {qa['n_annotated']} "
            f"({flagged}), discarded {discarded}{flag}")


# ------------------------------------------------------------- arguments

def add_common_args(parser, dataset=None):
    parser.set_defaults(_dataset=dataset)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpipe-source")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--segment", type=int, default=2560)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stanza-model-dir")
    parser.add_argument("--stanza-gpu", action="store_true")
    parser.add_argument("--stanza-package", default="default",
                        help="use the same package as the earlier silver parses if they must stay comparable")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--n", type=int, default=90,
                        help="alternatives per item per strategy. At 90 every one of the "
                             "6,435 pools is filled entirely from tier 1, complete and "
                             "clean first sentences: no incomplete text, no held-back "
                             "HTML / non-English, no padding anywhere. The binding pool is "
                             "naturalstories story2_s36/temp_075, whose last sentence makes "
                             "the model emit EOS for 403 of 500 raws, leaving 94 tier-1 draws")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="skip sentences with more Stanza words than this (0 = no limit)")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--no-annotation-cache", action="store_true",
                        help="annotate duplicate texts separately")
    # Comparability defaults. IV is a distance between the target and the
    # alternatives, so anything that treats the two differently becomes a bias
    # in the measurement. The defaults below are the comparable setting; each
    # can be switched back for a robustness run.
    parser.add_argument("--require-complete-sentence", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="skip a raw whose first sentence does not end in final punctuation; "
                             "the target is always a complete sentence, and the truncation rate "
                             "differs by strategy (2.4%% nucleus_08 ... 11.4%% temp_125). "
                             "Default: on.")
    parser.add_argument("--hold-back", action=argparse.BooleanOptionalAction, default=True,
                        help="defer HTML / non-English texts to the end of the pool. They are "
                             "ordinary samples from the model and were never needed to reach N, "
                             "so deferring them is equivalent to dropping them. Default: off.")
    parser.add_argument("--pad-to-n", action=argparse.BooleanOptionalAction, default=True,
                        help="always reach N: if every tier together falls short, repeat "
                             "kept draws in a cycle rather than leaving the document "
                             "incomplete. Copies are marked padding_duplicate. On by default")
    parser.add_argument("--incomplete-fallback", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="if complete first sentences cannot fill N, top up with "
                             "incomplete ones (the 120-token limit cut them off) rather "
                             "than leaving the document short. On by default")
    parser.add_argument("--lowercase-alternatives", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="fold alternatives to lower case. Default: on for CLASP, whose gold "
                             "context and targets are lower case; off for Natural Stories.")
    parser.add_argument("--context-annotation", choices=("manual", "corpipe"), default="manual",
                        help="manual (default): context entities from the hand annotation, which "
                             "anchors the cluster ids; corpipe: annotate the context with CorPipe "
                             "too, so nothing in the comparison is manual")
    parser.add_argument("--target-annotation", choices=("gold", "corpipe"), default="corpipe",
                        help="corpipe (default): annotate context, target and alternatives with the "
                             "same Stanza+seeded-CorPipe pass, so annotator error is shared and "
                             "cancels in the target-alternative distance; gold: manual entities for "
                             "the target only, which confounds the distance with the annotator "
                             "(robustness run)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete earlier results for the selected stories/IDs first")
    parser.add_argument("--keep-parts", action="store_true",
                        help="keep per-job part files after a document is assembled")
    parser.add_argument("--claim-dir", type=Path,
                        help="set by run_parallel.py: shared folder of job claims")
    parser.add_argument("--worker-name", default="main")


def check_args(parser, args):
    # Resolve the comparability defaults into the attribute names the rest of
    # the code reads. `no_hold_back` is kept because the settings block, the
    # part-file hash and the QA counters all refer to it by that name.
    args.no_hold_back = not args.hold_back
    if args.lowercase_alternatives is None:
        args.lowercase_alternatives = getattr(args, "_dataset", None) == "clasp"
    if min(args.n, args.batch_size, args.threads) <= 0 or args.max_tokens < 0:
        parser.error("N, batch size and threads must be positive; --max-tokens must be >= 0")
    if len(set(args.strategies)) != len(args.strategies):
        parser.error("duplicate requested strategies")
    if not args.corpipe_source:
        parser.error("--corpipe-source is required")
    if args.overwrite and args.claim_dir:
        parser.error("--overwrite with --claim-dir would let every worker delete the others' "
                     "results; run_parallel.py clears outputs once before starting workers")


def initialise(args):
    import spacy
    nlp = spacy.load(args.spacy_model)  # full pipeline, as in the IV reference code
    from corpipe26_seeded import SeededCorPipe
    # CorPipe/minnt must initialise before Stanza.
    annotator = SeededCorPipe(args.corpipe_source, args.model, segment=args.segment,
                             device=args.device, threads=args.threads)
    parser = StanzaUD(args.stanza_model_dir, args.stanza_gpu, args.stanza_package)
    return annotator, parser, nlp


# ------------------------------------------------------ parallel workers
#
# A job is one unit of work (Natural Stories: one sentence with all
# strategies; CLASP: one ID and one strategy). Each finished job is saved as
# a part file whose name carries a hash of the settings, so an existing file
# means "done with these settings". Workers take jobs in a fixed order
# (longest context first) and skip jobs that are done or claimed by another
# worker. A claim is a file created with O_EXCL, which is atomic.


def settings_hash(settings: dict) -> str:
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]


class WorkQueue:
    def __init__(self, claim_dir, worker):
        self.dir = Path(claim_dir) if claim_dir else None
        self.worker = worker
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def claim(self, key) -> bool:
        if self.dir is None:
            return True
        try:
            handle = os.open(self.dir / f"{key}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(handle, "w") as stream:
            stream.write(f"{self.worker} {socket.gethostname()} pid={os.getpid()} {time.strftime('%F %T')}\n")
        return True

    def release(self, key):
        if self.dir:
            (self.dir / f"{key}.claim").unlink(missing_ok=True)


def run_jobs(jobs, is_done, do_job, queue, max_failures_in_row=3):
    """Process every job that is neither done nor claimed. A job that raises
    keeps its claim (no other worker retries it in this run); a rerun of the
    launcher clears claims and retries it."""
    finished, failed, in_row = 0, [], 0
    for job in jobs:
        if is_done(job) or not queue.claim(job["key"]):
            continue
        if is_done(job):  # finished by another worker between the two checks
            queue.release(job["key"])
            continue
        started = time.time()
        try:
            do_job(job)
        except Exception:
            traceback.print_exc()
            print(f"[{queue.worker}] FAILED {job['key']}", flush=True)
            failed.append(job["key"])
            in_row += 1
            if in_row >= max_failures_in_row:
                print(f"[{queue.worker}] {in_row} failures in a row; stopping this worker", flush=True)
                break
            continue
        in_row = 0
        finished += 1
        queue.release(job["key"])
        print(f"[{queue.worker}] done {job['key']} in {time.time() - started:.0f}s", flush=True)
    return finished, failed


class LastUsed(dict):
    """Tiny LRU cache for per-document state inside a worker."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def get_or_make(self, key, make):
        if key in self:
            value = self.pop(key)
        else:
            value = make()
            while len(self) >= self.size:
                self.pop(next(iter(self)))
        self[key] = value
        return value
