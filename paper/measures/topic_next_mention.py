#!/usr/bin/env python3
"""Appendix A, "Topic": the topic as the entity a next-mention model fitted on GUM makes most expected.

The weight of an established entity is exp(theta . c(e)) with the context features of the full
next-mention account of train_next_mention.py (log gap since its last mention, gap = 1, log prior
mentions, information status given / accessible / new at its last mention; Givon, 1983); the topic is
the entity with the largest weight (ties: more mentions). The subject bit and first-mention bit of
the target and of every alternative are computed under this topic, as in topic.py, and the share of
alternatives that disagree with the target gives each bit's information value. Also written: whether
this topic is the most-mentioned entity (ties: the more recent one).

Usage: topic_next_mention.py <next-mention json> <gold annotation dir> <annotated json or directory> <out_csv>
"""

import json
import math
import sys

import numpy as np

from common import (alternatives_by_strategy, context_status, information_status, iter_records, mean,
                    mentions_of, record_key, target_sentences, write_csv)

CONTEXT_FEATURES = ("log_gap", "gap_prev", "log_prior_mentions", "status_MED", "status_NEW")


def entity_history(context_mentions, status_of, record):
    """entity -> (last context sentence, mentions, status at the last mention)."""
    n_context = len(context_mentions)
    history = {}
    for offset, mentions in enumerate(context_mentions):
        sentence_id = record["sentence_index"] - n_context + offset
        for mention in mentions:
            count = history.get(mention.entity, (0, 0, None))[1] + 1
            status = information_status(status_of.get((sentence_id, mention.entity), "new"))
            history[mention.entity] = (offset, count, status)
    return history


def next_mention_weights(history, n_context, theta):
    weights = {}
    for entity, (last, count, status) in history.items():
        gap = n_context - last
        features = np.array([math.log1p(gap), float(gap == 1), math.log1p(count),
                             float(status == "MED"), float(status == "NEW")])
        weights[entity] = float(np.exp(theta @ features))
    return weights


def topic_bits(sentences, topic, established):
    """(subject bit, first-mention bit) of an utterance for the given topic."""
    if topic is None:
        return 0, 0
    mentions = mentions_of(sentences)
    established_mentions = [m for m in mentions if m.entity in established]
    subject_bit = int(any(m.is_subject and m.entity == topic for m in mentions))
    first_bit = int(bool(established_mentions) and established_mentions[0].entity == topic)
    return subject_bit, first_bit


def main(model_json, gold_dir, inputs, out_csv):
    fits = json.load(open(model_json))
    theta = np.array(fits["accounts"]["full"]["theta"])[[fits["features"].index(f) for f in CONTEXT_FEATURES]]
    rows = []
    for record in iter_records([inputs]):
        context = record["context_sentences"]
        context_mentions = [mentions_of([sentence]) for sentence in context]
        history = entity_history(context_mentions, context_status(record, gold_dir), record)
        established = set(history)
        topic = topic_most_mentioned = None
        if established:
            weights = next_mention_weights(history, len(context), theta)
            topic = max(established, key=lambda e: (weights[e], history[e][1]))
            topic_most_mentioned = max(established, key=lambda e: (history[e][1], -history[e][0]))
        target = topic_bits(target_sentences(record), topic, established)
        base = record_key(record)
        base.update(n_established=len(established), same_topic_as_most_mentioned=int(topic == topic_most_mentioned),
                    nm_topic_is_subject=target[0], nm_topic_is_first=target[1])
        for strategy, alternatives in alternatives_by_strategy(record).items():
            described = [topic_bits(sentences, topic, established) for sentences in alternatives]
            rows.append(dict(base, strategy=strategy,
                             nm_s_subject=mean(abs(target[0] - a[0]) for a in described),
                             nm_s_first=mean(abs(target[1] - a[1]) for a in described)))
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
