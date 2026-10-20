#!/usr/bin/env python3
"""Next-mention models fitted on GUM, for Appendix A ("Claims tested and not reported above" and the
next-mention topic of the "Topic" paragraph).

(1) Which entity of an utterance U_m is mentioned first in U_{m+1}: a conditional logit over the
    entities of U_m, fitted under five accounts, each a restriction of the feature set:
      grammatical   most prominent grammatical function in U_m (subject is the reference level)
      recency       log gap to the entity's previous mention, gap = 1, discourse-new in U_m
      frequency     log prior mentions, log mentions in U_m
      infostatus    GUM information status (given / accessible / new) at U_m, discourse-new
      full          all of the above; "none" is the uniform expectation
    Five-fold cross-validation by document gives the held-out log2-likelihood per choice.
(2) The distribution of antecedent distance per referring form (same utterance, previous, 2-3,
    4-10, more than 10 utterances back, none), and the accessibility distance between two forms:
    the mean absolute difference of their cumulative distributions over the five finite bins.
(3) The marginal frequency of every referring form.

Usage: train_next_mention.py <GUM dep directory with GUM_*.conllu> <out json>
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict

import numpy as np

from common import (FUNCTION_RANK, GOLD_ENTITY, GRAMMATICAL_FUNCTIONS, REPO, grammatical_function,  # noqa: F401
                    information_status)
import build_gum_transition_cost as gum  # noqa: E402
from entity_roles import mention_head, mention_role  # noqa: E402
from form_inventory import FORMS, mention_form  # noqa: E402

FEATURES = ["f_subject_passive", "f_object_direct", "f_object_indirect", "f_oblique", "f_other", "log_mentions_in_U",
            "log_gap", "gap_prev", "discourse_new", "log_prior_mentions", "status_MED", "status_NEW"]
ACCOUNTS = {
    "none": [],
    "grammatical": ["f_subject_passive", "f_object_direct", "f_object_indirect", "f_oblique", "f_other"],
    "recency": ["log_gap", "gap_prev", "discourse_new"],
    "frequency": ["log_prior_mentions", "log_mentions_in_U"],
    "infostatus": ["status_MED", "status_NEW", "discourse_new"],
    "full": FEATURES,
}
DISTANCE_BINS = ("same", "prev", "2-3", "4-10", ">10", "none")
N_FOLDS = 5
L2_PENALTY = 1e-2


def distance_bin(gap):
    if gap == 0:
        return "same"
    if gap == 1:
        return "prev"
    if gap <= 3:
        return "2-3"
    if gap <= 10:
        return "4-10"
    return ">10"


def document_mentions(text):
    """Per sentence, the mentions of a GUM document: entity, information status, function, form."""
    sentences = gum.parse_conllu(text)
    out = []
    for tokens in sentences:
        mentions = []
        for start, end, entity in gum.entity_spans(tokens):
            status = "new"
            for match in GOLD_ENTITY.finditer(tokens[start]["misc"]):
                if int(match.group(1)) == entity:
                    status = match.group(3)
            head = mention_head(tokens, start, end)
            mentions.append({
                "entity": entity, "status": status,
                "function": grammatical_function(mention_role(tokens, head, "deprel")),
                "form": mention_form(tokens, start, end),
            })
        out.append(mentions)
    return out


def entity_features(mentions, history):
    """Feature vector of every entity mentioned in an utterance.
    history: entity -> (last sentence, prior mentions, gap) before this utterance."""
    summary = {}
    for mention in mentions:
        entry = summary.setdefault(mention["entity"], {"function": mention["function"], "n": 0, "status": mention["status"]})
        entry["n"] += 1
        if FUNCTION_RANK[mention["function"]] < FUNCTION_RANK[entry["function"]]:
            entry["function"] = mention["function"]
    vectors = {}
    for entity, entry in summary.items():
        new = entity not in history
        gap = None if new else history[entity][2]
        prior = 0 if new else history[entity][1]
        status = information_status(entry["status"])
        values = {name: 0.0 for name in FEATURES}
        if entry["function"] != "subject":
            values["f_" + entry["function"]] = 1.0
        values["log_mentions_in_U"] = math.log1p(entry["n"]) - math.log(2)
        values["log_gap"] = 0.0 if new else math.log1p(gap)
        values["gap_prev"] = 0.0 if new else float(gap == 1)
        values["discourse_new"] = float(new)
        values["log_prior_mentions"] = math.log1p(prior)
        values["status_MED"] = float(status == "MED")
        values["status_NEW"] = float(status == "NEW")
        vectors[entity] = np.array([values[name] for name in FEATURES])
    return vectors


def collect_instances(documents):
    """Choice instances (entities of U_m with the first referent of U_{m+1}) and antecedent-distance
    counts per form, over all documents."""
    choices = []
    antecedent_distance = defaultdict(Counter)
    for name, sentences in documents:
        last_and_count = {}                 # entity -> (last sentence, mentions so far)
        previous = None
        for index, mentions in enumerate(sentences):
            history = {e: (last, count, index - last) for e, (last, count) in last_and_count.items()}
            vectors = entity_features(mentions, history)
            for position, mention in enumerate(mentions):
                entity = mention["entity"]
                earlier_in_sentence = any(m["entity"] == entity for m in mentions[:position])
                if earlier_in_sentence:
                    bin_name = "same"
                elif entity in last_and_count:
                    bin_name = distance_bin(index - last_and_count[entity][0])
                else:
                    bin_name = "none"
                antecedent_distance[mention["form"]][bin_name] += 1
            if previous and mentions and mentions[0]["entity"] in previous:
                choices.append({"document": name, "candidates": previous, "chosen": mentions[0]["entity"]})
            for mention in mentions:
                entity = mention["entity"]
                last_and_count[entity] = (index, last_and_count.get(entity, (0, 0))[1] + 1)
            previous = vectors
    return choices, antecedent_distance


def fit_conditional_logit(instances, columns, iterations=30):
    """Newton's method for the conditional logit with an L2 penalty; theta over all FEATURES."""
    theta_full = np.zeros(len(FEATURES))
    if not columns:
        return theta_full
    theta = np.zeros(len(columns))
    for _ in range(iterations):
        gradient = L2_PENALTY * theta
        hessian = L2_PENALTY * np.eye(len(columns))
        for instance in instances:
            entities = list(instance["candidates"])
            design = np.array([instance["candidates"][e][columns] for e in entities])
            scores = design @ theta
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            chosen = np.array([float(e == instance["chosen"]) for e in entities])
            gradient += design.T @ (probabilities - chosen)
            expected = design.T @ probabilities
            hessian += (design * probabilities[:, None]).T @ design - np.outer(expected, expected)
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if np.abs(step).max() < 1e-6:
            break
    theta_full[columns] = theta
    return theta_full


def log2_likelihood_per_choice(instances, theta):
    total = 0.0
    for instance in instances:
        entities = list(instance["candidates"])
        scores = np.array([instance["candidates"][e] @ theta for e in entities])
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        total += math.log2(probabilities[entities.index(instance["chosen"])])
    return total / max(1, len(instances))


def main(gum_dir, out_json):
    documents = [(name, document_mentions(text)) for name, text in gum.iter_gum_documents(gum_dir)]
    choices, antecedent_distance = collect_instances(documents)
    names = sorted({name for name, _ in documents})
    random.seed(0)
    random.shuffle(names)
    fold_of = {name: index % N_FOLDS for index, name in enumerate(names)}
    print(f"GUM documents {len(documents)}; choice instances {len(choices)} "
          f"(mean choice set {np.mean([len(c['candidates']) for c in choices]):.2f})", flush=True)

    result = {"features": FEATURES, "accounts": {}}
    uniform_bits = float(np.mean([-math.log2(len(c["candidates"])) for c in choices]))
    for account, feature_names in ACCOUNTS.items():
        columns = [FEATURES.index(name) for name in feature_names]
        heldout = []
        for fold in range(N_FOLDS):
            train = [c for c in choices if fold_of[c["document"]] != fold]
            test = [c for c in choices if fold_of[c["document"]] == fold]
            heldout.append(log2_likelihood_per_choice(test, fit_conditional_logit(train, columns)))
        theta = fit_conditional_logit(choices, columns)
        result["accounts"][account] = {"theta": theta.tolist(), "heldout_bits_per_choice": float(np.mean(heldout)),
                                       "uniform_bits": uniform_bits}
        print(f"  {account:12s} held-out log2-likelihood per choice {np.mean(heldout):.4f} (uniform {uniform_bits:.4f})", flush=True)

    cumulative = {}
    for form in FORMS:
        counts = antecedent_distance.get(form, Counter())
        total = sum(counts.values())
        running = 0.0
        cumulative[form] = []
        for bin_name in DISTANCE_BINS:
            running += counts[bin_name] / total if total else 0.0
            cumulative[form].append(running)
    n_finite_bins = len(DISTANCE_BINS) - 1
    result["antecedent_distance"] = {"bins": DISTANCE_BINS, "cdf": cumulative,
                                     "counts": {form: dict(antecedent_distance.get(form, {})) for form in FORMS}}
    result["accessibility_distance"] = {
        a: {b: float(sum(abs(cumulative[a][i] - cumulative[b][i]) for i in range(n_finite_bins)) / n_finite_bins) for b in FORMS}
        for a in FORMS
    }
    result["form_counts"] = {form: sum(antecedent_distance.get(form, Counter()).values()) for form in FORMS}
    json.dump(result, open(out_json, "w"), indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
