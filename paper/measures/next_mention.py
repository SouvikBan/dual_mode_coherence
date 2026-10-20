#!/usr/bin/env python3
"""Appendix A, "Claims tested and not reported above", Natural Stories: next-mention expectations and
pronoun interpretation.

First referent. The entities of the previous utterance U_{n-1} form the choice set; each next-mention
account of train_next_mention.py gives a probability for every candidate (conditional logit with the
entity's features in U_{n-1}: grammatical function, recency, frequency, information status). For a
target whose first referent is one of these entities, surprisal_first_<account> is -log2 of that
probability and entropy_first_<account> the entropy of the distribution; log2 of the number of
candidates is the uniform expectation.

Pronoun interpretation (Kehler et al., 2008; Kehler & Rohde, 2013). For a target whose first mention is
a personal pronoun of an established entity, the candidates are the entities of the last two context
utterances that agree with the pronoun in number and gender. From the alternatives of a strategy, a
joint table n[e, a] counts those whose first referent is e with first form a (pronoun / other), with a
pseudo-count of one spread uniformly over the candidates and by the GUM form marginal over a. The
Expectancy model scores the target's referent by its next-mention probability pi_e, the Mirror model
by the production probability of a pronoun given e, the Bayesian model by their product normalised
over the candidates; each gives a surprisal in bits, next to the uniform log2 of the candidate count.

Usage: next_mention.py <next-mention json> <gold annotation dir> <annotated json or directory> <out_csv>
"""

import json
import math
import sys
from collections import Counter

import numpy as np

from common import (agreement_compatible, alternatives_by_strategy, context_status, information_status,
                    iter_records, mentions_of, per_entity, record_key, target_sentences, write_csv)

CANDIDATE_WINDOW = 2      # context utterances whose entities are candidate antecedents of a pronoun
PSEUDO_COUNT = 1.0


class NextMentionModel:
    def __init__(self, path):
        fits = json.load(open(path))
        self.features = fits["features"]
        self.theta = {account: np.array(values["theta"]) for account, values in fits["accounts"].items()}
        counts = fits["form_counts"]
        self.pronoun_share = counts["pron_personal"] / sum(counts.values())

    def feature_vector(self, entry, gap, prior_mentions, status):
        """Features of an entity of U_{n-1}: entry = its per_entity summary there; gap = utterances
        since its previous mention (None if new there); prior mentions before U_{n-1}; status at U_{n-1}."""
        new = gap is None
        values = {name: 0.0 for name in self.features}
        if entry["function"] != "subject":
            values["f_" + entry["function"]] = 1.0
        values["log_mentions_in_U"] = math.log1p(entry["n"]) - math.log(2)
        values["log_gap"] = 0.0 if new else math.log1p(gap)
        values["gap_prev"] = 0.0 if new else float(gap == 1)
        values["discourse_new"] = float(new)
        values["log_prior_mentions"] = math.log1p(prior_mentions)
        values["status_MED"] = float(status == "MED")
        values["status_NEW"] = float(status == "NEW")
        return np.array([values[name] for name in self.features])

    def distribution(self, account, candidate_vectors):
        entities = list(candidate_vectors)
        scores = np.array([self.theta[account] @ candidate_vectors[e] for e in entities])
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        return dict(zip(entities, probabilities))


def previous_utterance_candidates(context_mentions, status_of, record, model):
    """Feature vectors of the entities of U_{n-1}, with their history before U_{n-1}."""
    n_context = len(context_mentions)
    summary = per_entity(context_mentions[-1])
    candidates = {}
    for entity, entry in summary.items():
        earlier = [i for i, mentions in enumerate(context_mentions[:-1]) if any(m.entity == entity for m in mentions)]
        gap = None if not earlier else n_context - 1 - max(earlier)
        prior_mentions = sum(1 for mentions in context_mentions[:-1] for m in mentions if m.entity == entity)
        status = information_status(status_of.get((record["sentence_index"] - 1, entity), "new"))
        candidates[entity] = model.feature_vector(entry, gap, prior_mentions, status)
    return candidates


def context_agreement(context_mentions):
    """entity -> Number / Gender seen on its nominal or pronominal mentions in the context."""
    agreement = {}
    for mentions in context_mentions:
        for mention in mentions:
            if mention.upos in ("PRON", "PROPN", "NOUN"):
                entry = agreement.setdefault(mention.entity, {})
                for key, value in mention.agreement.items():
                    entry.setdefault(key, value)
    return agreement


def pronoun_models(first_referent, candidates, alternative_firsts, pronoun_share):
    """Surprisal of the target's referent under the Expectancy, Mirror and Bayesian models from the
    joint table of the alternatives' first referents and first forms."""
    table = Counter()
    for entity, is_pronoun in alternative_firsts:
        if entity in candidates:
            table[(entity, is_pronoun)] += 1
    prior = {(e, a): PSEUDO_COUNT * (pronoun_share if a else 1 - pronoun_share) / len(candidates)
             for e in candidates for a in (True, False)}
    next_mention = {e: table[(e, True)] + table[(e, False)] + prior[(e, True)] + prior[(e, False)] for e in candidates}
    pronoun_given = {e: (table[(e, True)] + prior[(e, True)]) / next_mention[e] for e in candidates}
    posterior = {e: table[(e, True)] + prior[(e, True)] for e in candidates}
    return {
        "surprisal_expectancy": -math.log2(next_mention[first_referent] / sum(next_mention.values())),
        "surprisal_mirror": -math.log2(pronoun_given[first_referent] / sum(pronoun_given.values())),
        "surprisal_bayesian": -math.log2(posterior[first_referent] / sum(posterior.values())),
        "n_alternatives_pronoun_candidates": sum(table[(e, True)] for e in candidates),
    }


def main(model_json, gold_dir, inputs, out_csv):
    model = NextMentionModel(model_json)
    rows = []
    for record in iter_records([inputs]):
        context = record["context_sentences"]
        context_mentions = [mentions_of([sentence]) for sentence in context]
        established = {m.entity for mentions in context_mentions for m in mentions}
        status_of = context_status(record, gold_dir)
        target_mentions = mentions_of(target_sentences(record))
        first = target_mentions[0] if target_mentions else None
        first_kind = "none" if first is None else ("established" if first.entity in established else "new")

        base = record_key(record)
        base.update(n_established=len(established), first_kind=first_kind)
        candidates = previous_utterance_candidates(context_mentions, status_of, record, model) if context else {}
        base["n_previous_entities"] = len(candidates)
        base["log2_M"] = math.log2(len(candidates)) if candidates else float("nan")
        base["first_in_previous"] = int(first is not None and first.entity in candidates)
        if candidates:
            for account in model.theta:
                probabilities = model.distribution(account, candidates)
                base[f"entropy_first_{account}"] = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)
                base[f"surprisal_first_{account}"] = (-math.log2(probabilities[first.entity])
                                                      if base["first_in_previous"] else float("nan"))

        pronoun_candidates = []
        if first is not None and first.form == "pron_personal" and first.entity in established:
            agreement = context_agreement(context_mentions)
            recent = {m.entity for mentions in context_mentions[-CANDIDATE_WINDOW:] for m in mentions}
            pronoun_candidates = sorted(e for e in recent if agreement_compatible(first.agreement, agreement.get(e, {})))
            if first.entity not in pronoun_candidates or len(pronoun_candidates) < 2:
                pronoun_candidates = []

        for strategy, alternatives in alternatives_by_strategy(record).items():
            row = dict(base, strategy=strategy)
            if pronoun_candidates:
                firsts = []
                for sentences in alternatives:
                    mentions = mentions_of(sentences)
                    firsts.append((mentions[0].entity, mentions[0].form == "pron_personal") if mentions else (None, None))
                row["pronoun_eligible"] = 1
                row["n_candidates"] = len(pronoun_candidates)
                row["surprisal_uniform"] = math.log2(len(pronoun_candidates))
                row["share_alternatives_outside_candidates"] = sum(1 for e, _ in firsts if e not in pronoun_candidates) / len(firsts)
                row.update(pronoun_models(first.entity, pronoun_candidates, firsts, model.pronoun_share))
            rows.append(row)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
