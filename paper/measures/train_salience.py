#!/usr/bin/env python3
"""Salience model for the topic bits of claim (iii) (paper, Eq. 9 and Appendix A, "Topic").

Trained on GUM (Lin & Zeldes, 2024 salience annotation): the target of an entity is the number
of the five human summaries of its document that mention it (0 to 5, the `s`/`n` block of the
Entity field in the misc column). Eight features describe how often and where the entity is
mentioned, all computable for an established entity from the context read so far:

    log(1 + mentions), share of sentences with a mention, relative position of the first and of
    the last mention, share of mentions as subject, share of mentions headed by a proper noun,
    mentioned in the first sentence, log(1 + tokens of the first mention).

A ridge regression on the standardised features predicts the graded score; five-fold
cross-validation by document reports the Spearman correlation with the score and the AUC for
the entities in a majority of summaries (the paper quotes the AUC, 0.88).

Usage: train_salience.py <GUM dep directory with GUM_*.conllu> <out json>
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from common import REPO  # noqa: F401  (puts scripts/entity_iv on sys.path)
import build_gum_transition_cost as gum  # noqa: E402

FEATURES = ["log_mentions", "sent_share", "first_rel", "last_rel",
            "subj_share", "proper_share", "in_first_sent", "log_first_len"]
ENTITY_OPEN = re.compile(r"\((\d+)-([a-z]+)-([a-z:]+)-([sn]+)-")
RIDGE_PENALTY = 1.0
N_FOLDS = 5


def head_of(sentence, start, end):
    """Token whose head lies outside the span (or the first token)."""
    for index in range(start, end + 1):
        head = sentence[index].get("head", "0")
        head = int(head) if str(head).isdigit() else 0
        if head == 0 or not (start + 1 <= head <= end + 1):
            return index
    return start


def doc_features(sentences, spans_per_sentence, proper_flags=None):
    """{entity: feature vector} over a document or a context.

    sentences: list of token lists (dicts with head, deprel, upos);
    spans_per_sentence: per sentence a list of (start, end, entity);
    proper_flags: optional {(sentence, start, end): True} for proper-name mentions when the head
    is not tagged PROPN (used for the CorPipe mention forms)."""
    n_sentences = len(sentences)
    stats = {}
    for sentence_index, (sentence, spans) in enumerate(zip(sentences, spans_per_sentence)):
        for start, end, entity in spans:
            head = head_of(sentence, start, end)
            deprel = str(sentence[head].get("deprel", "")).lower()
            upos = str(sentence[head].get("upos", ""))
            is_proper = upos == "PROPN" or bool(proper_flags and proper_flags.get((sentence_index, start, end), False))
            entry = stats.setdefault(entity, {
                "mentions": 0, "sentences": set(), "first": sentence_index, "last": sentence_index,
                "subject": 0, "proper": 0, "first_length": end - start + 1,
            })
            entry["mentions"] += 1
            entry["sentences"].add(sentence_index)
            entry["last"] = sentence_index
            entry["subject"] += int(deprel.startswith("nsubj") or deprel.startswith("csubj"))
            entry["proper"] += int(is_proper)
    features = {}
    for entity, entry in stats.items():
        features[entity] = np.array([
            np.log1p(entry["mentions"]),
            len(entry["sentences"]) / n_sentences,
            entry["first"] / max(n_sentences - 1, 1),
            entry["last"] / max(n_sentences - 1, 1),
            entry["subject"] / entry["mentions"],
            entry["proper"] / entry["mentions"],
            float(entry["first"] == 0),
            np.log1p(entry["first_length"]),
        ])
    return features


def salience_scores(sentences):
    """{entity: 'ssnns'-style summary block} from the misc column of a GUM document."""
    scores = {}
    for sentence in sentences:
        for token in sentence:
            for match in ENTITY_OPEN.finditer(token.get("misc", "")):
                scores.setdefault(int(match.group(1)), match.group(4))
    return scores


def ridge(features, target, penalty=RIDGE_PENALTY):
    """Ridge regression with an unpenalised intercept; returns [weights..., bias]."""
    design = np.hstack([features, np.ones((len(features), 1))])
    regulariser = penalty * np.diag(np.r_[np.ones(features.shape[1]), 0])
    return np.linalg.solve(design.T @ design + regulariser, design.T @ target)


def predict(theta, standardised):
    return np.hstack([standardised, np.ones((len(standardised), 1))]) @ theta


def auc(labels, scores):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_positive = labels.sum()
    n_negative = len(labels) - n_positive
    return (ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def spearman(a, b):
    rank = lambda values: np.argsort(np.argsort(values))
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def load_gum(gum_dir):
    rows, scores, majority, documents = [], [], [], []
    for path in sorted(Path(gum_dir).glob("GUM_*.conllu")):
        sentences = gum.parse_conllu(path.read_text(encoding="utf-8"))
        spans = [gum.entity_spans(sentence) for sentence in sentences]
        summary_blocks = salience_scores(sentences)
        for entity, vector in doc_features(sentences, spans).items():
            if entity not in summary_blocks:
                continue
            block = summary_blocks[entity]
            rows.append(vector)
            scores.append(float(block.count("s")))
            majority.append(float(block.count("s") / len(block) >= 0.5))
            documents.append(path.stem)
    return np.array(rows), np.array(scores), np.array(majority), np.array(documents)


def main(gum_dir, out_json):
    features, scores, majority, documents = load_gum(gum_dir)
    mu = features.mean(axis=0)
    sd = features.std(axis=0) + 1e-9
    standardised = (features - mu) / sd
    print("entities", len(scores), "| score distribution 0..5:",
          np.bincount(scores.astype(int), minlength=6).tolist(), "| documents", len(set(documents)))

    unique_documents = np.array(sorted(set(documents)))
    rng = np.random.default_rng(1)
    fold_of_document = {doc: i % N_FOLDS for i, doc in enumerate(rng.permutation(unique_documents))}
    folds = np.array([fold_of_document[doc] for doc in documents])
    aucs, rhos = [], []
    for fold in range(N_FOLDS):
        train, test = folds != fold, folds == fold
        theta = ridge(standardised[train], scores[train])
        predicted = predict(theta, standardised[test])
        aucs.append(auc(majority[test], predicted))
        rhos.append(spearman(predicted, scores[test]))
    print(f"{N_FOLDS}-fold by document: Spearman with the graded score",
          [round(r, 3) for r in rhos], "mean", round(float(np.mean(rhos)), 3),
          "| AUC for majority-salient entities", [round(a, 3) for a in aucs], "mean", round(float(np.mean(aucs)), 3))

    theta = ridge(standardised, scores)
    print("theta:", dict(zip(FEATURES + ["bias"], [round(float(v), 3) for v in theta])))
    json.dump({
        "features": FEATURES, "theta": theta.tolist(), "target": "graded salience 0-5, ridge",
        "mu": mu.tolist(), "sd": sd.tolist(), "cv_spearman": rhos, "cv_auc_majority": aucs,
        "n": int(len(scores)), "n_salient": int(majority.sum()),
    }, open(out_json, "w"), indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
