#!/usr/bin/env python3
"""Claim (iv): how new entities attach to established ones.

Entities are represented by the unit embedding of the head lemma of their first non-pronominal
mention (all-distilroberta-v1; fallback: the first mention). For an utterance U with new entities
N(U) and established entities E (paper, Eq. 10):

    beta(U) = sum_{e in N(U)} ( 1 - max_{e' in E} cos(v_e, v_e') )      (max over the empty set = 0)

Its information value compares the cost of the target with the costs of the alternatives
(Eq. 11, 2) and the anticipatory form compares the alternatives with each other (Eq. 3):

    d_beta         = mean_k | beta(T) - beta(A_k) |
    d_beta^A       = mean over ordered pairs k != l of | beta(A_k) - beta(A_l) |

Usage: linking_cost.py <annotated json or directory> <out_csv>
"""

import sys

import numpy as np

from common import (Embedder, alternatives_by_strategy, iter_records, mentions_of, record_key,
                    representative_lemmas, target_sentences, write_csv)


def new_entity_vectors(sentences, established, embedder):
    """Embeddings of the new entities of an utterance, one per entity."""
    lemmas = representative_lemmas(mentions_of(sentences))
    return [embedder.vector(lemma) for entity, lemma in lemmas.items() if entity not in established]


def linking_cost(new_vectors, anchor_matrix):
    """sum over new entities of 1 - max cosine to the anchors (cost 1 per entity without anchors)."""
    if not new_vectors:
        return 0.0
    if anchor_matrix is None:
        return float(len(new_vectors))
    return float(sum(1.0 - (anchor_matrix @ vector).max() for vector in new_vectors))


def main(inputs, out_csv):
    embedder = Embedder()
    rows = []
    for record in iter_records([inputs]):
        context_lemmas = representative_lemmas(mentions_of(record["context_sentences"]))
        established = set(context_lemmas)
        anchors = None
        if established:
            anchors = np.array([embedder.vector(lemma) for lemma in set(context_lemmas.values())])

        target_new = new_entity_vectors(target_sentences(record), established, embedder)
        beta_target = linking_cost(target_new, anchors)

        base = record_key(record)
        base.update(n_new=len(target_new), beta=beta_target)
        for strategy, alternatives in alternatives_by_strategy(record).items():
            beta_alternatives = np.array([
                linking_cost(new_entity_vectors(sentences, established, embedder), anchors)
                for sentences in alternatives
            ])
            k = len(beta_alternatives)
            pairwise = np.abs(beta_alternatives[:, None] - beta_alternatives[None, :])
            rows.append(dict(
                base,
                strategy=strategy,
                d_beta=float(np.abs(beta_target - beta_alternatives).mean()),
                d_beta_anticipatory=float(pairwise.sum() / (k * (k - 1))) if k > 1 else float("nan"),
                beta_alternatives_mean=float(beta_alternatives.mean()),
            ))
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
