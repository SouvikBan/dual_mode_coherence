#!/usr/bin/env python3
"""Claim (i): which established entities are mentioned.

For a target T and an alternative A, with J established entities (entities mentioned in the
context) and o_U(e) = 1 if utterance U mentions established entity e (paper, Eq. 4 and 5):

    d_est(T, A) = (1 / J) * sum_e | o_T(e) - o_A(e) |

The information value is the mean of d_est over the alternatives of a strategy (Eq. 2); the
anticipatory form is the mean over ordered pairs of distinct alternatives (Eq. 3). Both are 0 when
there is no established entity (story-initial utterances, 20 CLASP contexts).

Also written per target: the number of established entities, the number of them the target
mentions, and the number of new entities the target introduces.

Usage: established_entities.py <annotated json or directory> <out_csv>
"""

import sys

from common import (alternatives_by_strategy, iter_records, mean, mean_over_pairs, mentions_of,
                    record_key, target_sentences, write_csv)


def mentioned_entities(sentences):
    return {mention.entity for mention in mentions_of(sentences)}


def share_of_disagreement(mentioned_a, mentioned_b, established):
    """(1 / J) * sum over established entities of |o_a(e) - o_b(e)| = |symmetric difference| / J."""
    if not established:
        return 0.0
    return len((mentioned_a ^ mentioned_b) & established) / len(established)


def main(inputs, out_csv):
    rows = []
    for record in iter_records([inputs]):
        established = mentioned_entities(record["context_sentences"])
        target = mentioned_entities(target_sentences(record))
        base = record_key(record)
        base.update(
            n_established=len(established),
            n_mentioned_established=len(target & established),
            n_new=len(target - established),
        )
        for strategy, alternatives in alternatives_by_strategy(record).items():
            mentioned = [mentioned_entities(sentences) for sentences in alternatives]
            row = dict(base, strategy=strategy)
            row["d_est"] = mean(share_of_disagreement(target, alt, established) for alt in mentioned)
            row["d_est_anticipatory"] = mean_over_pairs(
                mentioned, lambda a, b: share_of_disagreement(a, b, established))
            rows.append(row)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
