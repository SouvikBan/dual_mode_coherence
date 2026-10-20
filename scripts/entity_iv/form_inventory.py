"""Referring-expression FORM of a mention, derived from its tokens.

GUM does not annotate form: its entity layer carries entity type, information
status, salience, identity and centering only. The pipeline's annotation stage
already derives a 4-way form (pronoun / proper / common / other) from the head
token's part of speech (corpipe26_seeded._mention_form). This module keeps that
idea but makes the inventory finer, using only information that exists both in
GUM's dep/*.conllu and in the pipeline's annotated JSON: the head token's
UPOS/XPOS, the first token of the span, and possessive marking inside the span.

Fine inventory (15 states, share of GUM mentions in parentheses):
    common_bare (21.8)  pron_personal (18.6)  common_definite (15.6)  proper (14.7)
    common_indefinite (8.0)  common_possessed (5.3)  pron_possessive (4.7)
    numeral (3.0)  pron_demonstrative (2.5)  verbal (1.9)  common_quantified (1.4)
    other (0.9)  pron_other (0.8)  adjectival (0.7)  pron_wh (0.1)

Head = the pipeline's own rule (entity_roles.mention_head), so a mention gets
the same form on both sides.
"""
from __future__ import annotations
from entity_roles import mention_head

FORMS = ("common_bare", "pron_personal", "common_definite", "proper", "common_indefinite",
         "common_possessed", "pron_possessive", "numeral", "pron_demonstrative", "verbal",
         "common_quantified", "other", "pron_other", "adjectival", "pron_wh")
COARSE_FORM = {"pron_personal": "pronoun", "pron_possessive": "pronoun", "pron_wh": "pronoun",
               "pron_demonstrative": "pronoun", "pron_other": "pronoun", "proper": "proper",
               "common_bare": "common", "common_definite": "common", "common_indefinite": "common",
               "common_possessed": "common", "common_quantified": "common",
               "numeral": "other", "verbal": "other", "adjectival": "other", "other": "other"}
DEF = {"the", "this", "that", "these", "those"}
INDEF = {"a", "an", "some", "any", "no", "another", "each", "every", "either", "neither", "such"}


def _s(token, key):
    return str(token.get(key, "") or "")


def mention_form(tokens, start: int, end: int) -> str:
    """Form of the mention spanning tokens[start..end] (inclusive)."""
    h = mention_head(tokens, start, end)
    head = tokens[h]
    xpos, upos, lemma = _s(head, "xpos"), _s(head, "upos"), _s(head, "lemma").lower()
    first = tokens[start]
    fl, fx = (_s(first, "lemma") or _s(first, "text")).lower(), _s(first, "xpos")
    if xpos == "PRP":
        return "pron_personal"
    if xpos == "PRP$":
        return "pron_possessive"
    if xpos in ("WP", "WP$", "WDT"):
        return "pron_wh"
    if xpos == "DT" or (upos == "PRON" and lemma in DEF):
        return "pron_demonstrative"
    if upos == "PRON":
        return "pron_other"
    if xpos in ("NNP", "NNPS") or upos == "PROPN":
        return "proper"
    if xpos in ("NN", "NNS") or upos == "NOUN":
        if start < h:
            possessed = any(_s(t, "xpos") == "PRP$" or _s(t, "deprel").split(":")[-1] == "poss"
                            for t in tokens[start:h])
            if possessed:
                return "common_possessed"
            if fl in DEF:
                return "common_definite"
            if fl in INDEF:
                return "common_indefinite"
            if fx == "CD":
                return "common_quantified"
        return "common_bare"
    if xpos == "CD":
        return "numeral"
    if upos in ("VERB", "AUX") or xpos.startswith("VB"):
        return "verbal"
    if upos == "ADJ":
        return "adjectival"
    return "other"
