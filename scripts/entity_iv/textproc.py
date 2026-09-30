#!/usr/bin/env python3
"""Small, deterministic English sentence/token adapter used by the smoke test."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_TOKEN_RE = re.compile(r"\w+(?:[’']\w+)*|[^\w\s]", re.UNICODE)
_BOUNDARY_RE = re.compile(r"(?<=[.!?…])(?:[\"'’”\])}]*)\s+(?=\S)")
_FINAL_PUNCT = tuple(".!?…") + tuple("\"'’”)]};:")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    lemma: str = "_"
    upos: str = "X"
    xpos: str = "_"
    head: int = 0
    deprel: str = "dep"


class TextProcessor:
    """Use spaCy when installed, with a stdlib fallback for dry-run tests.

    CorPipe itself only consumes the token forms. POS/dependencies are retained in
    CoNLL-U for inspection, but they are not used by the seeded inference patch.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.backend = "regex"
        self.nlp = None
        try:
            import spacy  # type: ignore

            try:
                self.nlp = spacy.load(spacy_model)
                self.backend = f"spacy:{spacy_model}"
            except OSError:
                self.nlp = spacy.blank("en")
                self.backend = "spacy:blank_en"
            if not any(name in self.nlp.pipe_names for name in ("parser", "senter", "sentencizer")):
                self.nlp.add_pipe("sentencizer")
        except ImportError:
            pass

    def sentences(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self.nlp is not None:
            return [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]
        starts = [0]
        for match in _BOUNDARY_RE.finditer(text):
            starts.append(match.end())
        out = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            sent = text[start:end].strip()
            if sent:
                out.append(sent)
        return out

    def first_sentence(self, text: str, char_cap: int = 1500) -> tuple[str | None, bool]:
        text = (text or "").strip()
        if not text:
            return None, False
        sents = self.sentences(text[:char_cap])
        if not sents:
            return None, False
        first = sents[0]
        complete = len(sents) >= 2 or first.rstrip().endswith(_FINAL_PUNCT)
        return first, complete

    def tokens(self, sentence: str) -> list[Token]:
        if self.nlp is None:
            return [Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(sentence)]

        doc = self.nlp(sentence)
        parsed = "parser" in self.nlp.pipe_names
        out = []
        for token in doc:
            if token.is_space:
                continue
            head = token.head.i + 1 if parsed and token.head.i != token.i else 0
            deprel = token.dep_ if parsed and token.dep_ else ("root" if head == 0 else "dep")
            out.append(Token(
                text=token.text,
                start=token.idx,
                end=token.idx + len(token.text),
                lemma=token.lemma_ or "_",
                upos=token.pos_ or "X",
                xpos=token.tag_ or "_",
                head=head,
                deprel=deprel,
            ))
        return out


def normalise(text: str) -> str:
    return " ".join((text or "").lower().split()).rstrip(".!?,;:")


def join_sentences(sentences: Iterable[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())
