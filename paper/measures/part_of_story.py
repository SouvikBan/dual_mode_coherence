#!/usr/bin/env python3
"""Claim (ii): which part of the story is continued.

Each established entity e is represented by the unit embedding v_e of the head lemma of its first
non-pronominal mention in the context (all-distilroberta-v1). An utterance U is represented by the
mean embedding of the established entities it mentions (paper, Eq. 6),

    mu_U = (1 / |R(U)|) * sum_{e in R(U)} v_e ,

on an extra coordinate when R(U) is empty (footnote 1), and two utterances are compared by
(Eq. 7)

    d_part(T, A) = 1/2 * || mu_T - mu_A ||^2 .

With K alternatives and mu_bar their mean, the information value and its anticipatory form
(Eq. 2 and 3) are computed in closed form:

    d_part        = mean_k 1/2 ||mu_T - mu_k||^2 = 1/2 (||mu_T||^2 + mean_k ||mu_k||^2 - 2 mu_T . mu_bar)
    d_part^A      = mean_{k,l} 1/2 ||mu_k - mu_l||^2 = mean_k ||mu_k||^2 - ||mu_bar||^2   (all K^2 pairs)

The variant of Appendix A that represents an entity by the earlier utterances it appeared in
(a unit-normalised 0/1 vector over the context sentences instead of v_e) is written as
d_part_history and d_part_history_anticipatory, with the same formulas.

Usage: part_of_story.py <annotated json or directory> <out_csv>
"""

import sys

import numpy as np

from common import (EMBEDDING_DIM, Embedder, alternatives_by_strategy, entities_of, iter_records,
                    mentions_of, record_key, representative_lemmas, target_sentences, write_csv)


def with_empty_coordinate(vector, dim):
    """Append the extra coordinate: 0 for a mean vector, 1 (and zeros) for the empty set."""
    out = np.zeros(dim + 1)
    if vector is None:
        out[dim] = 1.0
    else:
        out[:dim] = vector
    return out


def utterance_vector(sentences, established_vectors, dim):
    """mu_U over the established entities that the utterance mentions."""
    entities = entities_of(mentions_of(sentences))
    returning = [entity for entity in entities if entity in established_vectors]
    if not returning:
        return with_empty_coordinate(None, dim), 0
    vectors = np.array([established_vectors[entity] for entity in returning])
    return with_empty_coordinate(vectors.mean(axis=0), dim), len(returning)


def information_values(target_vector, alternative_vectors):
    """(responsive, anticipatory) squared-distance information values in closed form."""
    matrix = np.array(alternative_vectors)
    mean_vector = matrix.mean(axis=0)
    squared_norms = (matrix * matrix).sum(axis=1)
    responsive = 0.5 * (target_vector @ target_vector + squared_norms.mean() - 2 * target_vector @ mean_vector)
    anticipatory = squared_norms.mean() - mean_vector @ mean_vector
    return float(responsive), float(anticipatory)


def history_vectors(context_sentences):
    """entity -> unit-normalised occurrence vector over the context sentences (Appendix A variant)."""
    n_sentences = len(context_sentences)
    occurrence = {}
    for sentence_index, sentence in enumerate(context_sentences):
        for mention in mentions_of([sentence]):
            occurrence.setdefault(mention.entity, np.zeros(n_sentences))[sentence_index] = 1.0
    return {entity: vector / np.linalg.norm(vector) for entity, vector in occurrence.items()}


def main(inputs, out_csv):
    embedder = Embedder()
    rows = []
    for record in iter_records([inputs]):
        context = record["context_sentences"]
        lemmas = representative_lemmas(mentions_of(context))
        semantic = {entity: embedder.vector(lemma) for entity, lemma in lemmas.items()}
        history = history_vectors(context)
        n_context = len(context)

        target = target_sentences(record)
        target_semantic, n_returning = utterance_vector(target, semantic, EMBEDDING_DIM)
        target_history, _ = utterance_vector(target, history, n_context)

        base = record_key(record)
        base.update(n_established=len(semantic), n_mentioned_established=n_returning)
        for strategy, alternatives in alternatives_by_strategy(record).items():
            row = dict(base, strategy=strategy)
            alt_semantic = [utterance_vector(s, semantic, EMBEDDING_DIM)[0] for s in alternatives]
            row["d_part"], row["d_part_anticipatory"] = information_values(target_semantic, alt_semantic)
            alt_history = [utterance_vector(s, history, n_context)[0] for s in alternatives]
            row["d_part_history"], row["d_part_history_anticipatory"] = information_values(target_history, alt_history)
            rows.append(row)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
