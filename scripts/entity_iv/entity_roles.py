#!/usr/bin/env python3
"""One definition of a mention's grammatical role, shared by every script.

The role mapping used to be copied into calculate_entity_iv.py,
build_gum_transition_cost.py, count_gold_entities.py and
entity_transition_model.py. Keeping four copies is how the definitions drifted,
so they all import from here now.

Two inventories
---------------
``coarse`` reproduces the original four states, kept so earlier results stay
reproducible:

    subject, object, oblique, other

It has two known problems. ``relation_base`` stripped subtypes, so
``nmod:poss`` fell together with ``nmod`` and then with ``obl``; measured on
10,272 real mention heads only 37% of the "oblique" bin was genuinely oblique,
36% of it being possessives, which are 14.2% of all mentions. And ``appos`` was
not inherited although ``conj`` was, so appositives dropped to "other".

``fine`` (the default) separates the distinctions that the coarse inventory
conflated, and inherits through ``appos`` as well as ``conj``:

    subject            nsubj, csubj, nsubj:outer            31.9% of mentions
    subject_passive    nsubj:pass, csubj:pass                2.2%
    object_direct      obj                                  12.8%
    object_indirect    iobj                                  1.0%
    possessor          nmod:poss, det:poss                   14.2%
    oblique            obl and its non-agent subtypes       17.5%
    oblique_agent      obl:agent                             0.6%
    nominal_modifier   nmod and its non-possessive subtypes  9.5%
    compound           compound, flat, amod, nummod          1.2%
    predicate          root, and a copula's predicate        1.8%
    clausal            ccomp, xcomp, advcl, acl, acl:relcl   1.1%
    other              everything else                       ~1%

A mention headed by ``conj`` or ``appos`` takes the role of what it is joined
or apposed to, following the chain up until a different relation is reached.
"""

from __future__ import annotations

from typing import Sequence

# The entity grid of Barzilay & Lapata (2005, 2008) represents each entity in
# each sentence as one of S (subject), O (object), X (present but neither) or
# "-" (absent), and defines coherence over transitions between adjacent cells.
# Centering (Grosz, Joshi & Weinstein 1995; Brennan, Friedman & Pollard 1987)
# ranks the forward-looking centres by grammatical function, standardly
# SUBJECT > OBJECT > OTHER, and Guinaudeau & Strube (2013) weight S/O/X as
# 3/2/1 with absence 0. Two consequences for this project:
#
#   * absence is a state of the representation, not a missing value. That is
#     the same point the sentence-level transition model makes.
#   * a possessive is neither subject nor object, so it belongs in X. Putting
#     it with the verbal obliques, as the original four-state mapping did, has
#     no basis in this literature.
#
# GRID_ROLES is that canonical three-way distinction, for comparability with
# published work. FINE_ROLES refines it without crossing its boundaries: every
# fine role rolls up into exactly one grid role via TO_GRID, so a fine-grained
# model can always be collapsed back to S/O/X.
GRID_ROLES = ("subject", "object", "other")
COARSE_ROLES = ("subject", "object", "oblique", "other")
FINE_ROLES = ("subject", "subject_passive", "object_direct", "object_indirect",
              "possessor", "oblique", "oblique_agent", "nominal_modifier",
              "compound", "predicate", "clausal", "other")
# "deprel": the state IS the Stanza label of the mention-span head, so the
# inventory is exhaustive by construction and nothing is put in a bucket by
# hand. The state list is therefore not fixed here: it is read off the training
# corpus when a cost model is fitted and stored in that model, and a label not
# seen at fit time projects to "other" (see project_state).
DEPREL_INVENTORY = "deprel"
INVENTORIES = {"grid": GRID_ROLES, "coarse": COARSE_ROLES, "fine": FINE_ROLES,
               DEPREL_INVENTORY: ()}
INHERIT_DEFAULT = {"deprel": ("conj", "appos")}

# Every fine role belongs to exactly one entity-grid category.
TO_GRID = {"subject": "subject", "subject_passive": "subject",
           "object_direct": "object", "object_indirect": "object",
           "possessor": "other", "oblique": "other", "oblique_agent": "other",
           "nominal_modifier": "other", "compound": "other", "predicate": "other",
           "clausal": "other", "other": "other"}
_GRID = {"nsubj": "subject", "csubj": "subject",
         "obj": "object", "iobj": "object"}

# Relations a mention inherits through: it is not itself an argument, it shares
# the position of the thing it is attached to. The original mapping inherited
# only through conj, which left appositives in "other"; that is corrected in
# the fine inventory, while coarse keeps the old behaviour so earlier results
# stay reproducible.
INHERITED = {"grid": ("conj", "appos"), "coarse": ("conj",), "fine": ("conj", "appos"),
             "deprel": ("conj", "appos")}

# Universal Dependencies version 1 spellings, so a gold-v2 robustness run and a
# Stanza run land in the same states.
V1_ALIASES = {"dobj": "obj", "nsubjpass": "nsubj:pass", "csubjpass": "csubj:pass",
              "auxpass": "aux:pass", "poss": "nmod:poss", "name": "flat"}

# Stanza's English model and GUM share 48 of ~51 labels, because Stanza's
# default package (combined_charlm) is trained on the combined UD English
# treebanks, GUM among them. The one convention that has moved is npmod, which
# UD English renamed to unmarked; Stanza still emits the old spelling, so
# without this a Stanza obl:npmod would not match a GUM obl:unmarked state.
SUBTYPE_ALIASES = {"npmod": "unmarked", "tmod": "unmarked"}

_COARSE = {"nsubj": "subject", "csubj": "subject",
           "obj": "object", "iobj": "object", "ccomp": "object", "xcomp": "object",
           "obl": "oblique", "nmod": "oblique"}

# Full label first, then the bare relation; so nmod:poss and nmod differ while
# an unlisted subtype still falls back to its base relation.
_FINE_FULL = {"nsubj:pass": "subject_passive", "csubj:pass": "subject_passive",
              "nsubj:outer": "subject", "csubj:outer": "subject",
              "nmod:poss": "possessor", "det:poss": "possessor",
              "obl:agent": "oblique_agent",
              "acl:relcl": "clausal"}
_FINE_BASE = {"nsubj": "subject", "csubj": "subject",
              "obj": "object_direct", "iobj": "object_indirect",
              "nmod": "nominal_modifier", "obl": "oblique",
              "compound": "compound", "flat": "compound", "amod": "compound",
              "nummod": "compound",
              "root": "predicate",
              "ccomp": "clausal", "xcomp": "clausal", "advcl": "clausal",
              "acl": "clausal"}


def normalise_deprel(value: object) -> str:
    """Lower-cased label with version 1 spellings mapped to version 2."""
    label = str(value or "_").strip().lower()
    base, _, subtype = label.partition(":")
    base = V1_ALIASES.get(base, base)
    subtype = SUBTYPE_ALIASES.get(subtype, subtype)
    if ":" in base:                      # an alias that already carries a subtype
        return base if not subtype else f"{base.split(':')[0]}:{subtype}"
    return f"{base}:{subtype}" if subtype else base


def relation_base(value: object) -> str:
    return normalise_deprel(value).split(":", 1)[0]


def deprel_states(counts, min_count: int = 1) -> list[str]:
    """State list for the deprel inventory: every label seen at least min_count
    times, plus "other" if anything was folded away."""
    keep = sorted(label for label, n in counts.items() if n >= min_count)
    if any(n < min_count for n in counts.values()) or "other" not in keep:
        keep = [label for label in keep if label != "other"] + ["other"]
    return keep


def project_state(label: str, states) -> str:
    """Map a label onto a fitted state list; anything unseen becomes "other"."""
    if label in states:
        return label
    base = label.split(":", 1)[0]
    return base if base in states else "other"


def role_of_deprel(value: object, inventory: str = "fine") -> str:
    label = normalise_deprel(value)
    if inventory == "deprel":
        return label
    if inventory == "grid":
        return _GRID.get(label.split(":", 1)[0], "other")
    if inventory == "coarse":
        return _COARSE.get(label.split(":", 1)[0], "other")
    if label in _FINE_FULL:
        return _FINE_FULL[label]
    return _FINE_BASE.get(label.split(":", 1)[0], "other")


def mention_head(tokens: Sequence[dict], start: int, end: int) -> int:
    """Index of the mention's syntactic head: the first non-punctuation token in
    the span whose governor lies outside it."""
    if not 0 <= start <= end < len(tokens):
        raise ValueError(f"invalid mention span ({start}, {end})")
    span_ids = {int(tokens[i].get("id", i + 1)) for i in range(start, end + 1)}
    candidates = [i for i in range(start, end + 1)
                  if int(tokens[i].get("head", 0)) not in span_ids] or list(range(start, end + 1))
    content = [i for i in candidates
               if str(tokens[i].get("upos", "")).upper() != "PUNCT"
               and str(tokens[i].get("xpos", "")) not in {".", ",", ":", "-LRB-", "-RRB-"}]
    return (content or candidates)[0]


def inherited_relation(tokens: Sequence[dict], head_index: int,
                       inventory: str = "fine") -> str:
    """Label of the mention head, following conj (and, except for the coarse
    inventory, appos) up to their anchor."""
    inherited = INHERITED[inventory]
    by_id = {int(token.get("id", index + 1)): index for index, token in enumerate(tokens)}
    seen: set[int] = set()
    index = head_index
    while index not in seen:
        seen.add(index)
        label = normalise_deprel(tokens[index].get("deprel"))
        if label.split(":", 1)[0] not in inherited:
            return label
        governor = int(tokens[index].get("head", 0))
        if governor == 0 or governor not in by_id:
            break
        index = by_id[governor]
    return "other"


def mention_role(tokens: Sequence[dict], head_index: int, inventory: str = "fine") -> str:
    """Role of the mention whose head is at head_index."""
    label = inherited_relation(tokens, head_index, inventory)
    if label == "other":
        return "other"
    role = role_of_deprel(label, inventory)
    if role == "predicate" and inventory == "fine":
        return "predicate"
    return role


def role_of_mention(tokens: Sequence[dict], mention: dict, inventory: str = "fine",
                    use_minspan: bool = False) -> str:
    """Role of a mention given as a dict with token_start / token_end."""
    start, end = int(mention["token_start"]), int(mention["token_end"])
    if use_minspan:
        from build_gum_inf_transition_cost import minspan_range
        start, end = minspan_range(mention.get("minspan"), start, end)
    return mention_role(tokens, mention_head(tokens, start, end), inventory)
