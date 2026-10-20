"""Additional entity IV channels for calculate_entity_iv.py.

Transition channels (same comparison rule as calculate_entity_iv.transition_distance: the
entities that recur from the context in the target or the alternative; an absent entity costs 0;
Soergel distance sum|c_t - c_a| / sum max(c_t, c_a)); what changes is the cost c of the entity's
first mention in the continuation:

    d3_transition_form       additively smoothed bigram over referring-expression FORM (15 states,
                             form_inventory.FORMS), P(form | previous form), fitted on GUM
    d3_transition_role_mlp   one-hidden-layer MLP P(role | previous role, previous form, gap, #mentions),
                             fine role inventory (12 states), fitted on GUM
    d3_transition_form_mlp   the same MLP for FORM

Entity-graph channel (Guinaudeau & Strube 2013):

    d8_graph                 the context is a sentence x entity matrix B with weights 3 (subject),
                             2 (object), 1 (other). A continuation's connection profile over the
                             context sentences is g = B b, b = its role-weighted vector over context
                             entities. Distance = 1 - cos(g_target, g_alternative); 0 if both are empty,
                             1 if exactly one is empty.

All four are computed from the annotated JSON only (no embeddings). The models come from
build_gum_models.py (gum_transition_models.json).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from build_gum_models import FeatureSpec, mlp_log2_probs
from entity_roles import INVENTORIES, mention_head, mention_role, project_state
from form_inventory import mention_form

EXT_METRICS = ("d3_transition_form", "d3_transition_role_mlp", "d3_transition_form_mlp", "d8_graph")
TRANSITION_METRICS = EXT_METRICS[:3]
RANK = {"subject": 3, "object": 2, "other": 1}


def coarse_role(role: str) -> str:
    r = role.lower()
    if "subj" in r:
        return "subject"
    if r in ("obj", "iobj") or r.startswith("obj"):
        return "object"
    return "other"


def branch_mentions(sentences: Sequence[dict]) -> list[tuple[int, int, str, str, str]]:
    """[(sentence_index, cluster_id, fine role, form, coarse role)] in document order."""
    out = []
    for s_idx, sentence in enumerate(sentences):
        tokens = sentence["tokens"]
        mentions = sorted(sentence.get("mentions", []),
                          key=lambda m: (int(m["token_start"]), -int(m["token_end"]), int(m["cluster_id"])))
        for m in mentions:
            start, end = int(m["token_start"]), int(m["token_end"])
            h = mention_head(tokens, start, end)
            fine = project_state(mention_role(tokens, h, "fine"), INVENTORIES["fine"])
            out.append((s_idx, int(m["cluster_id"]), fine, mention_form(tokens, start, end),
                        coarse_role(mention_role(tokens, h, "deprel"))))
    return out


class ContextState:
    """Everything about a context that the extra channels need; built once per item."""

    def __init__(self, context_sentences: Sequence[dict]):
        ms = branch_mentions(context_sentences)
        self.n_sentences = len(context_sentences)
        self.last: dict[int, tuple[str, str, int, int]] = {}   # eid -> (fine role, form, last sentence, n mentions)
        count: dict[int, int] = defaultdict(int)
        self.B: dict[tuple[int, int], float] = defaultdict(float)  # (sentence, eid) -> weight
        for s_idx, eid, fine, form, coarse in ms:
            count[eid] += 1
            self.last[eid] = (fine, form, s_idx, count[eid])
            self.B[(s_idx, eid)] = max(self.B[(s_idx, eid)], RANK[coarse])
        self.entities = set(self.last)


class Continuation:
    """First mentions (for the transition channels) and the graph profile of one continuation."""

    def __init__(self, sentences: Sequence[dict], context: ContextState):
        ms = branch_mentions(sentences)
        self.first: dict[int, tuple[str, str]] = {}
        b: dict[int, float] = defaultdict(float)
        for _s, eid, fine, form, coarse in ms:
            self.first.setdefault(eid, (fine, form))
            if eid in context.entities:
                b[eid] = max(b[eid], RANK[coarse])
        g = [0.0] * context.n_sentences
        for (s_idx, eid), w in context.B.items():
            if eid in b:
                g[s_idx] += w * b[eid]
        self.profile = g


class TransitionModels:
    def __init__(self, models: dict):
        ch = models["channels"]
        for name in ("form_bigram", "role_mlp", "form_mlp"):
            if name not in ch:
                raise ValueError(f"transition model file has no '{name}' channel")
        self.form_costs = ch["form_bigram"]["transition_costs"]
        self.role_mlp = ch["role_mlp"]["weights"]
        self.form_mlp = ch["form_mlp"]["weights"]
        self.spec = FeatureSpec(models["roles"], models["forms"], tuple(ch["role_mlp"]["features"]["inputs"]))
        self.role_ix = {s: i for i, s in enumerate(models["roles"])}
        self.form_ix = {s: i for i, s in enumerate(models["forms"])}
        self.roles, self.forms = list(models["roles"]), list(models["forms"])

    def costs(self, previous: tuple[str, str, int, int], gap: int, current: tuple[str, str]) -> dict[str, float]:
        p_role, p_form, _s, n_prev = previous
        x = self.spec.vector(p_role, p_form, gap, n_prev)[None, :]
        lp_role = mlp_log2_probs(self.role_mlp, x)[0]
        lp_form = mlp_log2_probs(self.form_mlp, x)[0]
        return {
            "d3_transition_form": float(self.form_costs[p_form][current[1]]),
            "d3_transition_role_mlp": float(-lp_role[self.role_ix[current[0]]]),
            "d3_transition_form_mlp": float(-lp_form[self.form_ix[current[1]]]),
        }


def transition_distances(context: ContextState, target: Continuation, alternative: Continuation,
                         models: TransitionModels) -> dict[str, float]:
    recurrent = sorted(context.entities & (set(target.first) | set(alternative.first)))
    zero = {m: 0.0 for m in TRANSITION_METRICS}
    num = dict(zero); den = dict(zero)
    for eid in recurrent:
        previous = context.last[eid]
        gap = context.n_sentences - previous[2]
        ct = models.costs(previous, gap, target.first[eid]) if eid in target.first else zero
        ca = models.costs(previous, gap, alternative.first[eid]) if eid in alternative.first else zero
        for m in TRANSITION_METRICS:
            num[m] += abs(ct[m] - ca[m])
            den[m] += max(ct[m], ca[m])
    return {m: (num[m] / den[m] if den[m] else 0.0) for m in TRANSITION_METRICS}


def graph_distance(target: Continuation, alternative: Continuation) -> float:
    x, y = target.profile, alternative.profile
    nx = math.sqrt(sum(v * v for v in x)); ny = math.sqrt(sum(v * v for v in y))
    if nx == 0 and ny == 0:
        return 0.0
    if nx == 0 or ny == 0:
        return 1.0
    return 1 - sum(a * b for a, b in zip(x, y)) / (nx * ny)


class ExtScorer:
    """Per-comparison extra channels with a per-item cache of the context and target state."""

    def __init__(self, models: dict | None):
        self.models = TransitionModels(models) if models is not None else None
        self._cache_key = None
        self._context = None
        self._target = None

    @property
    def metrics(self) -> tuple[str, ...]:
        return EXT_METRICS if self.models is not None else ("d8_graph",)

    def row(self, comparison) -> dict[str, float]:
        key = (comparison.dataset, comparison.document_id, comparison.item_id, comparison.sentence_index, comparison.language)
        if key != self._cache_key:
            from calculate_entity_iv import branch_sentences
            self._context = ContextState(comparison.context_sentences)
            self._target = Continuation(branch_sentences(comparison.target), self._context)
            self._cache_key = key
        from calculate_entity_iv import branch_sentences
        alternative = Continuation(branch_sentences(comparison.alternative), self._context)
        out = {"d8_graph": graph_distance(self._target, alternative)}
        if self.models is not None:
            out.update(transition_distances(self._context, self._target, alternative, self.models))
        return out
