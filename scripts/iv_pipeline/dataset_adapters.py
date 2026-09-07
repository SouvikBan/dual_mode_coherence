#!/usr/bin/env python3
"""Canonical Natural Stories/CLASP loaders and silver UD annotation."""

from __future__ import annotations

import ast
import csv
import difflib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from naturalstories_conllx import (
    iv_role,
    load_naturalstories_conllx,
)


def normalise_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _token_offsets(text: str, forms: Sequence[str],
                   ptb_surface: bool = False) -> list[tuple[int, int]]:
    """Align treebank tokens to an exact surface string without retokenizing it."""
    if ptb_surface:
        raise ValueError("PTB surface conversion is not used by this pipeline")
    surface_forms = list(forms)
    compact_text_chars, compact_to_text = [], []
    for index, char in enumerate(text):
        if not char.isspace():
            compact_text_chars.append(char)
            compact_to_text.append(index)
    compact_text = "".join(compact_text_chars)
    compact_forms = "".join(surface_forms)
    compare_table = str.maketrans({
        '"': "'", "‘": "'", "’": "'", "“": "'", "”": "'",
        "–": "-", "—": "-",
    })
    source = compact_forms.translate(compare_table)
    target = compact_text.translate(compare_table)
    boundary_map = list(range(len(source) + 1))
    if source != target:
        boundary_map = [0] * (len(source) + 1)
        matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            source_width, target_width = i2 - i1, j2 - j1
            if source_width == 0:
                continue
            for offset in range(source_width + 1):
                boundary_map[i1 + offset] = (
                    j1 + round(offset * target_width / source_width)
                )
        boundary_map[0], boundary_map[-1] = 0, len(target)
        for index in range(1, len(boundary_map)):
            boundary_map[index] = max(boundary_map[index], boundary_map[index - 1])
    offsets, cursor = [], 0
    for form in surface_forms:
        source_start, source_end = cursor, cursor + len(form)
        start_compact, end_compact = boundary_map[source_start], boundary_map[source_end]
        if end_compact <= start_compact:
            raise ValueError(f"could not align token {form!r} in {text!r}")
        start = compact_to_text[start_compact]
        end = compact_to_text[end_compact - 1] + 1
        offsets.append((start, end))
        cursor = source_end
    return offsets


def load_naturalstories_gold(conllx_path: str | Path) -> dict[str, list[dict]]:
    """Return the same CoNLL-X-derived sentences used for LM generation.

    No second corpus surface file is consulted. This prevents sentence targets
    and cumulative contexts from drifting away from the surprisal split.
    """

    return load_naturalstories_conllx(conllx_path)


def parse_ratings(value: str) -> list[int]:
    parsed = ast.literal_eval(f"[{value}]") if value.strip() else []
    if not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"non-integer CLASP ratings: {value!r}")
    return parsed


def load_clasp_gold(path: str | Path) -> dict[int, list[dict]]:
    """Load canonical five-target CLASP groups, including human ratings."""
    groups: dict[int, list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            context_id = int(row["ID"])
            groups[context_id].append({
                "row_index": row_index,
                "context_id": context_id,
                "language": row["Language"],
                "sentence": normalise_text(row["Sentence"]),
                "pre_context": normalise_text(row["Pre-Context"]),
                "post_context": normalise_text(row["Post-Context"]),
                "ratings_without_context": parse_ratings(row["Without-Context Ratings"]),
                "ratings_with_context": parse_ratings(row["With-Context Ratings"]),
            })
    expected = ["English", "Czech", "German", "Spanish", "French"]
    for context_id, rows in groups.items():
        languages = [row["language"] for row in rows]
        if languages != expected:
            raise ValueError(
                f"CLASP ID {context_id} languages/order {languages}, expected {expected}"
            )
        if len({row["pre_context"] for row in rows}) != 1:
            raise ValueError(f"CLASP ID {context_id} does not share one pre-context")
    return dict(groups)


class StanzaSilverParser:
    """English UD parser for CLASP and generated alternatives."""

    def __init__(self, model_dir: str | None = None, use_gpu: bool = False) -> None:
        try:
            import stanza  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "stanza is required for silver grammar: pip install stanza; "
                "python -c \"import stanza; stanza.download('en')\""
            ) from exc
        kwargs = {
            "lang": "en",
            "processors": "tokenize,mwt,pos,lemma,depparse",
            "tokenize_no_ssplit": True,
            "use_gpu": use_gpu,
            "verbose": False,
        }
        if model_dir:
            kwargs["dir"] = model_dir
        self.pipeline = stanza.Pipeline(**kwargs)

    def parse(self, text: str) -> dict:
        text = normalise_text(text)
        document = self.pipeline(text)
        if len(document.sentences) != 1:
            raise ValueError(
                f"silver parser returned {len(document.sentences)} sentences with "
                "tokenize_no_ssplit=True"
            )
        sentence = document.sentences[0]
        tokens = []
        for word in sentence.words:
            start = getattr(word, "start_char", None)
            end = getattr(word, "end_char", None)
            tokens.append({
                "id": int(word.id),
                "text": word.text,
                "lemma": word.lemma or "_",
                "upos": word.upos or "X",
                "xpos": word.xpos or "_",
                "feats": word.feats or "_",
                "head": int(word.head),
                "deprel": word.deprel or "dep",
                "deps": word.deps or "_",
                "misc": word.misc or "_",
                "char_start": start,
                "char_end": end,
                "grammar_source": "stanza_en_ud_silver",
                "grammatical_role": iv_role(word.deprel),
            })
        # Always realign Stanza forms to our exact normalized surface. Stanza
        # occasionally returns unreliable offsets around repeated apostrophes.
        # Its token ``''`` is literal text, unlike PTB ``''`` in the Natural
        # Stories treebank, which represents a closing double quote.
        offsets = _token_offsets(text, [token["text"] for token in tokens])
        for token, (start, end) in zip(tokens, offsets):
            token["char_start"], token["char_end"] = start, end
        return {
            "text": text,
            "tokens": tokens,
            "grammar_source": "stanza_en_ud_silver",
        }

    def parse_many(self, texts: Iterable[str]) -> list[dict]:
        return [self.parse(text) for text in texts]


def validate_sentence_spec(sentence: dict) -> None:
    text, tokens = sentence["text"], sentence["tokens"]
    if not tokens:
        raise ValueError("grammatical sentence has no tokens")
    ids = [int(token["id"]) for token in tokens]
    if ids != list(range(1, len(tokens) + 1)):
        raise ValueError(f"token ids are not contiguous: {ids}")
    for token in tokens:
        start, end = int(token["char_start"]), int(token["char_end"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"bad character offsets {(start, end)} in {text!r}")
        surface = text[start:end]
        expected = token["text"]
        if normalise_text(surface) != normalise_text(expected):
            raise ValueError(
                f"token offset mismatch: token={token['text']!r}, "
                f"expected={expected!r}, surface={surface!r}, "
                f"source={token.get('grammar_source')!r}"
            )
