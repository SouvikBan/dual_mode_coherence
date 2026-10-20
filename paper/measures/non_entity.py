#!/usr/bin/env python3
"""The ten non-entity measures (paper, Section 4).

From Giulianelli et al. (2023): k-gram distances over words and part-of-speech tags, k = 1, 2, 3,
and the cosine distance between sentence embeddings; added here: k-gram distances over dependency
labels. A k-gram distance is 1 - |A and B| / |A or B| over the k-gram multisets of the two
utterances (0 when both are empty). The sentence embeddings come from all-distilroberta-v1.

Each is used as d in Eq. (2); the mean over ordered pairs of alternatives (Eq. 3) is written as
the *_anticipatory column.

Usage: non_entity.py <annotated json or directory> <out_csv>
"""

import sys
from collections import Counter

import numpy as np

from common import (Embedder, alternatives_by_strategy, iter_records, mean, mean_over_pairs,
                    record_key, target_sentences, write_csv)

KGRAM_LAYERS = {"lex": "text", "pos": "xpos", "dep": "deprel"}
ORDERS = (1, 2, 3)


def kgrams(sequence, k):
    return Counter(tuple(sequence[i:i + k]) for i in range(len(sequence) - k + 1))


def kgram_distance(a, b):
    if not a and not b:
        return 0.0
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return 1 - intersection / union if union else 0.0


def features(sentences):
    tokens = [token for sentence in sentences for token in sentence["tokens"]]
    sequences = {
        "lex": [token["text"].lower() for token in tokens],
        "pos": [token.get("xpos") or token["upos"] for token in tokens],
        "dep": [token["deprel"] for token in tokens],
    }
    out = {f"{layer}{k}": kgrams(sequence, k) for layer, sequence in sequences.items() for k in ORDERS}
    out["text"] = " ".join(token["text"] for token in tokens)
    return out


def main(inputs, out_csv):
    embedder = Embedder()
    rows = []
    for record in iter_records([inputs]):
        target = features(target_sentences(record))
        for strategy, alternatives in alternatives_by_strategy(record).items():
            alternative_features = [features(sentences) for sentences in alternatives]
            row = dict(record_key(record), strategy=strategy)
            for layer in KGRAM_LAYERS:
                for k in ORDERS:
                    name = f"{layer}{k}"
                    row[f"d_{name}"] = mean(kgram_distance(target[name], alt[name]) for alt in alternative_features)
                    row[f"d_{name}_anticipatory"] = mean_over_pairs(
                        alternative_features, lambda a, b: kgram_distance(a[name], b[name]))
            embeddings = embedder.sentence_vectors([target["text"]] + [alt["text"] for alt in alternative_features])
            similarity = embeddings @ embeddings.T
            n_alternatives = len(alternative_features)
            row["d_cos"] = float(np.mean(1 - similarity[0, 1:]))
            upper = np.triu_indices(n_alternatives, 1)
            row["d_cos_anticipatory"] = float((1 - similarity[1:, 1:])[upper].mean())
            rows.append(row)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
