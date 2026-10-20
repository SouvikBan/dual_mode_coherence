#!/usr/bin/env python3
"""Appendix A, "Claims tested and not reported above": a distance over referring forms.

Two orderings of referring forms: the Givenness Hierarchy (Gundel et al., 1993), as ranks
pronoun 1, demonstrative 2, definite / possessed / proper 4, bare / indefinite 6, scaled to [0, 1] by
the largest difference; and the typical antecedent distance (Ariel, 1990), as the accessibility
distance between two forms fitted on GUM by train_next_mention.py (mean absolute difference of their
antecedent-distance distributions).

For every established entity the target mentions, its form in the target is compared with its form
in each alternative that also mentions it; the information value is the mean over those
alternatives, averaged over the target's established entities. The first-referent version compares
only the form of the target's first referent, over the alternatives whose first referent it is.

Usage: referring_forms.py <next-mention json> <annotated json or directory> <out_csv>
"""

import json
import sys

from common import (alternatives_by_strategy, iter_records, mean, mentions_of, per_entity, record_key,
                    target_sentences, write_csv)

GIVENNESS_RANK = {"pron_personal": 1, "pron_possessive": 1, "pron_demonstrative": 2,
                  "common_definite": 4, "common_possessed": 4, "proper": 4,
                  "common_bare": 6, "common_indefinite": 6}
GIVENNESS_RANGE = 5.0


def givenness_distance(form_a, form_b):
    if form_a in GIVENNESS_RANK and form_b in GIVENNESS_RANK:
        return abs(GIVENNESS_RANK[form_a] - GIVENNESS_RANK[form_b]) / GIVENNESS_RANGE
    return float("nan")


def defined(values):
    return [v for v in values if v == v]


def main(model_json, inputs, out_csv):
    accessibility = json.load(open(model_json))["accessibility_distance"]
    rows = []
    for record in iter_records([inputs]):
        established = {m.entity for sentence in record["context_sentences"] for m in mentions_of([sentence])}
        target_mentions = mentions_of(target_sentences(record))
        target = per_entity(target_mentions)
        target_established = [e for e in target if e in established]
        first = target_mentions[0].entity if target_mentions else None
        base = record_key(record)
        base.update(n_established=len(established), n_mentioned_established=len(target_established))
        for strategy, alternatives in alternatives_by_strategy(record).items():
            described = []
            for sentences in alternatives:
                mentions = mentions_of(sentences)
                described.append({"entities": per_entity(mentions), "first": mentions[0].entity if mentions else None})
            row = dict(base, strategy=strategy)
            accessibility_values, givenness_values, n_eligible = [], [], []
            for entity in target_established:
                form = target[entity]["form"]
                eligible = [a["entities"][entity]["form"] for a in described if entity in a["entities"]]
                n_eligible.append(len(eligible))
                if not eligible:
                    continue
                accessibility_values.append(mean(accessibility[form][other] for other in eligible))
                ranks = defined(givenness_distance(form, other) for other in eligible)
                if ranks:
                    givenness_values.append(mean(ranks))
            row["d_form_accessibility"] = mean(accessibility_values)
            row["d_form_givenness"] = mean(givenness_values)
            row["n_eligible_alternatives_mean"] = mean(n_eligible)
            if first in established:
                form = target[first]["form"]
                eligible = [a["entities"][first]["form"] for a in described if a["first"] == first]
                if eligible:
                    row["d_form_first_accessibility"] = mean(accessibility[form][other] for other in eligible)
                    row["d_form_first_givenness"] = mean(defined(givenness_distance(form, other) for other in eligible))
                    row["n_eligible_first"] = len(eligible)
            rows.append(row)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
