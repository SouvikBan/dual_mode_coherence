#!/usr/bin/env python3
"""Claim (iii): whether the topic is realized again as the subject.

Subject distance (paper, Eq. 8): an utterance is described by the entity of its first subject
(nsubj / csubj, active or passive), represented by the unit embedding of its head lemma (for an
established entity, the lemma of its first non-pronominal context mention); an utterance without a
subject sits on an extra coordinate. d_top(T, A) = 1 - cos(v_s(T), v_s(A)); the information value
is the mean over alternatives, the anticipatory form the mean over ordered pairs of alternatives.

Topic bits (Eq. 9): the topic e* is the established entity that the GUM salience model
(train_salience.py) rates highest (ties: more mentions). s_U = 1[the subject of U is e*],
phi_U = 1[the first established entity U mentions is e*]. Their information values are the share of
alternatives that disagree with the target on the bit. The same bits under the most-mentioned
entity as topic are written for comparison (Appendix A), with whether the two topics coincide.

Usage: topic.py <salience model json> <annotated json or directory> <out_csv>
"""

import json
import sys
from collections import Counter

import numpy as np

from common import (EMBEDDING_DIM, Embedder, alternatives_by_strategy, iter_records, mean,
                    mentions_of, record_key, representative_lemmas, target_sentences, write_csv)
from train_salience import doc_features

NO_SUBJECT = np.zeros(EMBEDDING_DIM + 1)
NO_SUBJECT[EMBEDDING_DIM] = 1.0


class SalienceModel:
    def __init__(self, path):
        model = json.load(open(path))
        self.theta = np.array(model["theta"])
        self.mu = np.array(model["mu"])
        self.sd = np.array(model["sd"])

    def predict(self, feature_vector):
        return float(np.r_[(feature_vector - self.mu) / self.sd, 1.0] @ self.theta)


def context_tables(context_sentences):
    """Token tables, mention spans and proper-name flags of the context, as doc_features wants them."""
    tokens, spans, proper = [], [], {}
    for sentence_index, sentence in enumerate(context_sentences):
        tokens.append([{"head": str(t.get("head", 0)), "deprel": t.get("deprel", ""), "upos": t.get("upos", "")}
                       for t in sentence["tokens"]])
        sentence_spans = []
        for mention in mentions_of([sentence]):
            sentence_spans.append((mention.token_start, mention.token_end, mention.entity))
            proper[(sentence_index, mention.token_start, mention.token_end)] = mention.form == "proper"
        spans.append(sorted(sentence_spans))
    return tokens, spans, proper


def topics_of_context(context_sentences, salience):
    """(topic by learned salience, topic by mention count, established entities)."""
    if not context_sentences:
        return None, None, set()
    tokens, spans, proper = context_tables(context_sentences)
    features = doc_features(tokens, spans, proper)
    counts = Counter(entity for sentence_spans in spans for _, _, entity in sentence_spans)
    established = set(features)
    if not established:
        return None, None, established
    scores = {entity: salience.predict(vector) for entity, vector in features.items()}
    topic_salience = max(established, key=lambda entity: (scores[entity], counts[entity]))
    topic_count = counts.most_common(1)[0][0]
    return topic_salience, topic_count, established


def describe(sentences, topic, established, context_lemmas, embedder):
    """(subject bit, first-mention bit, subject vector) of an utterance for the given topic."""
    if topic is None:
        return 0, 0, NO_SUBJECT
    mentions = mentions_of(sentences)
    established_mentions = [m for m in mentions if m.entity in established]
    subjects = [m for m in mentions if m.is_subject]
    subject_bit = int(any(m.entity == topic for m in subjects))
    first_bit = int(bool(established_mentions) and established_mentions[0].entity == topic)
    if not subjects:
        return subject_bit, first_bit, NO_SUBJECT
    subject = subjects[0]
    lemma = context_lemmas.get(subject.entity, subject.lemma)
    vector = np.zeros(EMBEDDING_DIM + 1)
    vector[:EMBEDDING_DIM] = embedder.vector(lemma)
    return subject_bit, first_bit, vector


def main(model_json, inputs, out_csv):
    salience = SalienceModel(model_json)
    embedder = Embedder()
    rows = []
    for record in iter_records([inputs]):
        context = record["context_sentences"]
        topic_salience, topic_count, established = topics_of_context(context, salience)
        context_lemmas = representative_lemmas(mentions_of(context))
        target = target_sentences(record)
        target_by_topic = {
            "salience": describe(target, topic_salience, established, context_lemmas, embedder),
            "count": describe(target, topic_count, established, context_lemmas, embedder),
        }
        base = record_key(record)
        base.update(
            n_established=len(established),
            same_topic=int(topic_count == topic_salience),
            topic_is_subject=target_by_topic["salience"][0],
            topic_is_first=target_by_topic["salience"][1],
            count_topic_is_subject=target_by_topic["count"][0],
            count_topic_is_first=target_by_topic["count"][1],
        )
        for strategy, alternatives in alternatives_by_strategy(record).items():
            row = dict(base, strategy=strategy)
            described = {
                "salience": [describe(s, topic_salience, established, context_lemmas, embedder) for s in alternatives],
                "count": [describe(s, topic_count, established, context_lemmas, embedder) for s in alternatives],
            }
            # topic bits: share of alternatives that disagree with the target
            row["s_subject"] = mean(abs(target_by_topic["salience"][0] - a[0]) for a in described["salience"])
            row["s_first"] = mean(abs(target_by_topic["salience"][1] - a[1]) for a in described["salience"])
            row["count_s_subject"] = mean(abs(target_by_topic["count"][0] - a[0]) for a in described["count"])
            row["count_s_first"] = mean(abs(target_by_topic["count"][1] - a[1]) for a in described["count"])
            row["share_alternatives_topic_subject"] = mean(a[0] for a in described["salience"])
            # subject distance: 1 - cosine between subject vectors (topic-free)
            subject_vectors = np.array([a[2] for a in described["salience"]])
            k = len(subject_vectors)
            gram = subject_vectors @ subject_vectors.T
            row["d_top"] = float(np.mean(1.0 - subject_vectors @ target_by_topic["salience"][2]))
            row["d_top_anticipatory"] = float((k * k - gram.sum()) / (k * (k - 1))) if k > 1 else float("nan")
            rows.append(row)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
