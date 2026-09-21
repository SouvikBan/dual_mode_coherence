#!/usr/bin/env python3
"""Entity, lexical, POS-overlap, dependency-label and semantic IV per strategy.

Use exactly N annotated alternatives for each item/strategy (default 100).
Write only mean, minimum and 80th percentile, separately for CLASP and NS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


METRICS = (
    "d1_recurrence_uniform",
    "d1_recurrence_exponential",
    "d2_new_head",
    "d3_transition",
    "d_lexical_1gram",
    "d_lexical_2gram",
    "d_lexical_3gram",
    "d_pos_overlap_1gram",
    "d_pos_overlap_2gram",
    "d_pos_overlap_3gram",
    "d_syntactic_dependency_1gram",
    "d_syntactic_dependency_2gram",
    "d_syntactic_dependency_3gram",
    "d_semantic_cosine",
    "d_semantic_euclidean",
)
STRATEGIES = (
    "ancestral", "temp_075", "temp_125", "nucleus_08", "nucleus_085",
    "nucleus_09", "nucleus_095", "typical_02", "typical_03", "typical_085", "typical_095",
)
from entity_roles import (INVENTORIES, mention_head, mention_role,  # noqa: E402
                          project_state, relation_base as role_relation_base)

ROLES = INVENTORIES["fine"]   # replaced from --role-inventory in main()
ROLE_INVENTORY = "fine"
UD_RELATIONS = set("acl advcl advmod amod appos aux case cc ccomp clf compound conj cop csubj dep det discourse dislocated expl fixed flat goeswith iobj list mark nmod nsubj nummod obj obl orphan parataxis punct reparandum root vocative xcomp".split())


@dataclass(frozen=True)
class MentionFeature:
    key: tuple[int, int, int, int]
    sentence_index: int
    token_start: int
    token_end: int
    cluster_id: int
    head_index: int
    head_surface: str
    role: str


@dataclass
class BranchFeatures:
    mentions: list[MentionFeature]
    clusters: dict[int, list[MentionFeature]]


@dataclass
class Comparison:
    dataset: str
    document_id: str
    item_id: str
    sentence_index: int
    language: str
    strategy: str
    alternative_id: str
    sample_index: int | str
    context_sentences: list[dict]
    target: dict
    alternative: dict


def normalised_surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(value.strip().split()) or "_"


def branch_text(branch: dict) -> str:
    """Return the original branch text, reconstructing it only if necessary."""

    text = branch.get("text")
    if isinstance(text, str):
        return text.strip()
    return " ".join(
        str(sentence.get("text", "")).strip()
        for sentence in branch.get("sentences", [])
        if str(sentence.get("text", "")).strip()
    )


class TextDistanceScorer:
    """Reference-compatible lexical, POS, and sentence-embedding distances."""

    def __init__(
        self,
        spacy_model: str,
        semantic_model: str,
        device: str,
        spacy_batch_size: int,
        embedding_batch_size: int,
    ) -> None:
        import spacy
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.spacy_batch_size = spacy_batch_size
        self.embedding_batch_size = embedding_batch_size
        self.nlp = spacy.load(
            spacy_model,
            disable=["parser", "ner", "lemmatizer"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(semantic_model)
        self.model = AutoModel.from_pretrained(semantic_model).eval().to(self.device)
        self.lexical: dict[str, list[str]] = {}
        self.pos_overlap: dict[str, list[str]] = {}
        self.embeddings: dict[str, object] = {}

    @staticmethod
    def make_ngrams(values: Sequence[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(values[index:index + n]) for index in range(len(values) - n + 1)]

    @staticmethod
    def inverse_overlap(
        left: Sequence[tuple[str, ...]],
        right: Sequence[tuple[str, ...]],
    ) -> float:
        """Implement scorer.py's symmetric inverse-overlap distance."""

        denominator = len(left) + len(right)
        if denominator == 0:
            return 0.0
        left_types, right_types = set(left), set(right)
        shared = sum(value in left_types for value in right)
        shared += sum(value in right_types for value in left)
        return (denominator - shared) / denominator

    def prepare(self, texts: Sequence[str]) -> None:
        """Parse and embed every unique target/alternative text in batches."""

        self.lexical.clear()
        self.pos_overlap.clear()
        self.embeddings.clear()
        unique = list(dict.fromkeys(texts))
        if any(not text for text in unique):
            raise ValueError("empty target/alternative text")
        print(f"preparing text distances for {len(unique)} unique utterances", flush=True)
        for text, doc in zip(
            unique,
            self.nlp.pipe(unique, batch_size=self.spacy_batch_size),
        ):
            self.lexical[text] = [token.text.lower() for token in doc]
            self.pos_overlap[text] = [token.pos_ for token in doc]

        for start in range(0, len(unique), self.embedding_batch_size):
            batch = unique[start:start + self.embedding_batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            for text, embedding in zip(batch, pooled.cpu()):
                if not self.torch.isfinite(embedding).all() or float(embedding.norm()) < 1e-8:
                    raise ValueError(f"invalid semantic embedding for {text!r}")
                self.embeddings[text] = embedding
            completed = min(start + len(batch), len(unique))
            batch_number = start // self.embedding_batch_size + 1
            if completed == len(unique) or batch_number % 100 == 0:
                print(f"embedded {completed}/{len(unique)} utterances", flush=True)

    def distances(self, target: str, alternative: str) -> dict[str, float]:
        output: dict[str, float] = {}
        for n in range(1, 4):
            target_words = self.make_ngrams(self.lexical[target], n)
            alternative_words = self.make_ngrams(self.lexical[alternative], n)
            output[f"d_lexical_{n}gram"] = self.inverse_overlap(
                target_words, alternative_words
            )

            target_pos = self.make_ngrams(self.pos_overlap[target], n)
            alternative_pos = self.make_ngrams(self.pos_overlap[alternative], n)
            output[f"d_pos_overlap_{n}gram"] = self.inverse_overlap(
                target_pos, alternative_pos
            )

        target_embedding = self.embeddings[target]
        alternative_embedding = self.embeddings[alternative]
        cosine_similarity = float(self.torch.dot(target_embedding, alternative_embedding))
        output["d_semantic_cosine"] = 1.0 - max(-1.0, min(1.0, cosine_similarity))
        output["d_semantic_euclidean"] = float(
            self.torch.linalg.vector_norm(target_embedding - alternative_embedding)
        )
        return output


def relation_base(value: str) -> str:
    relation = (value or "_").casefold()
    aliases = {
        "nsubjpass": "nsubj",
        "csubjpass": "csubj",
        "dobj": "obj",
    }
    base = relation.split(":", 1)[0]
    return aliases.get(base, base)


def dependency_label(token):
    label = token.get("deprel")
    if not isinstance(label, str) or label in ("", "_"):
        raise ValueError("missing DEPREL; regenerate/repair the annotation")
    # Compare the basic UD relation, including root and punct. These are label
    # aliases only: they do not convert Stanford/CoNLL-X trees into UD trees.
    base = relation_base(label)
    base = "aux" if base == "auxpass" else base
    if base not in UD_RELATIONS:
        raise ValueError(f"non-UD dependency label {label!r}; gold and silver must use compatible dependency schemes")
    return base


def dependency_ngrams(branch, n):
    grams = []
    for sentence in branch_sentences(branch):
        tokens = sentence["tokens"]
        if not tokens:
            raise ValueError("empty annotated sentence")
        labels = [dependency_label(token) for token in tokens]
        grams.extend(TextDistanceScorer.make_ngrams(labels, n))
    return grams


def dependency_distances(target, alternative):
    return {
        f"d_syntactic_dependency_{n}gram": TextDistanceScorer.inverse_overlap(
            dependency_ngrams(target, n), dependency_ngrams(alternative, n)
        ) for n in range(1, 4)
    }


def validate_transition_costs(costs):
    for previous in ROLES:
        for current in ROLES:
            try:
                value = float(costs[previous][current])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"missing/invalid transition cost {previous}->{current}") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"transition cost must be finite and nonnegative: {previous}->{current}")
    if not any(float(costs[a][b]) > 0 for a in ROLES for b in ROLES):
        raise ValueError("all transition costs are zero; d3 would be uninformative")
    if any(float(costs[a][b]) == 0 for a in ROLES for b in ROLES):
        print("Note: zero transition costs make that transition indistinguishable from absence in d3.", flush=True)


def branch_features(sentences: Sequence[dict]) -> BranchFeatures:
    features = []
    ordinal = 0
    for sentence_index, sentence in enumerate(sentences):
        tokens = sentence.get("tokens")
        if not tokens or not isinstance(sentence.get("mentions"), list):
            raise ValueError("missing tokens/entity annotation; zero entities requires mentions=[]")
        ids = {int(token["id"]) for token in tokens}
        if len(ids) != len(tokens):
            raise ValueError("duplicate token IDs")
        for token in tokens:
            if not token.get("text") or int(token["head"]) not in ids | {0}:
                raise ValueError("missing word or invalid head")
            dependency_label(token)
        mentions = sorted(
            sentence.get("mentions", []),
            key=lambda mention: (
                int(mention["token_start"]),
                -int(mention["token_end"]),
                int(mention["cluster_id"]),
            ),
        )
        for mention in mentions:
            start, end = int(mention["token_start"]), int(mention["token_end"])
            if not 0 <= start <= end < len(tokens):
                raise ValueError(f"invalid mention span {(start, end)}")
            head_index = mention_head(tokens, start, end)
            feature = MentionFeature(
                key=(sentence_index, start, end, ordinal),
                sentence_index=sentence_index,
                token_start=start,
                token_end=end,
                cluster_id=int(mention["cluster_id"]),
                head_index=head_index,
                head_surface=normalised_surface(tokens[head_index].get("text", "")),
                role=project_state(mention_role(tokens, head_index, ROLE_INVENTORY), ROLES),
            )
            features.append(feature)
            ordinal += 1
    clusters: dict[int, list[MentionFeature]] = defaultdict(list)
    for mention in features:
        clusters[mention.cluster_id].append(mention)
    return BranchFeatures(features, dict(clusters))


def branch_sentences(branch: dict) -> list[dict]:
    if not isinstance(branch, dict) or not isinstance(branch.get("sentences"), list) or not branch["sentences"]:
        raise ValueError("branch must contain a nonempty sentences list")
    return branch["sentences"]


def iter_annotation_files(inputs: Sequence[str | Path]) -> Iterable[Path]:
    seen = set()
    for value in inputs:
        path = Path(value)
        candidates = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def iter_item_records(
    inputs: Sequence[str | Path],
    dataset_filter: str = "auto",
) -> Iterable[dict]:
    """Read each target and its complete annotated alternative set."""

    for path in iter_annotation_files(inputs):
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        dataset = document.get("dataset")
        if dataset not in {"naturalstories", "clasp"}:
            continue
        if dataset_filter != "auto" and dataset != dataset_filter:
            continue
        if not document.get("complete") or not document.get("items"):
            raise ValueError(f"{path}: incomplete annotation document")
        full_context = document["context"]
        document_id = str(document.get("story_id") or document.get("context_id") or path.stem)
        shared_alternatives = document.get("alternatives") or document.get("shared_alternatives")
        for item in document.get("items", []):
            prefix_count = int(item.get("prefix_sentence_count", item.get("sentence_index", len(full_context))))
            context = full_context[:prefix_count]
            target = item.get("target")
            alternatives = item.get("alternatives")
            if not isinstance(alternatives, list):
                alternatives = shared_alternatives
            # Backward compatibility with the earlier Natural Stories layout.
            if (not isinstance(target, dict) or not isinstance(alternatives, list)) and item.get("branches"):
                branches = item["branches"]
                target = branches[0]
                alternatives = branches[1:]
            if not isinstance(target, dict) or not isinstance(alternatives, list):
                raise ValueError(f"{path}: item missing target/alternatives")
            if not 0 <= prefix_count <= len(full_context):
                raise ValueError(f"{path}: invalid prefix length")
            language = str(item.get("language") or target.get("language") or "English")
            yield {
                "dataset": dataset,
                "document_id": document_id,
                "item_id": str(item.get("id", "")),
                "sentence_index": int(item.get("sentence_index", prefix_count)),
                "language": language,
                "context_sentences": context,
                "target": target,
                "alternatives": alternatives,
            }


def alternatives_by_strategy(alternatives: Sequence[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for alternative in alternatives:
        if not alternative.get("strategy") or not alternative.get("sentences"):
            raise ValueError("alternative missing strategy or annotated sentences")
        groups[alternative["strategy"]].append(alternative)
    return dict(groups)


def choose_strategy_sample_sizes(inputs, dataset_filter, requested, expected_strategies):
    """Every requested strategy must have exactly N samples at every item."""
    sizes, seen = {}, set()
    for record in iter_item_records(inputs, dataset_filter):
        identity = (record["dataset"], record["document_id"], record["item_id"])
        if identity in seen:
            raise ValueError(f"duplicate annotated item: {identity}")
        seen.add(identity)
        groups = alternatives_by_strategy(record["alternatives"])
        for strategy in expected_strategies:
            alternatives = groups.get(strategy, [])
            if len(alternatives) != requested:
                raise ValueError(f"{identity}/{strategy}: expected exactly {requested}, found {len(alternatives)}")
            ids = [a.get("id") for a in alternatives]
            if None in ids or "" in ids or len(set(ids)) != len(ids):
                raise ValueError(f"{identity}/{strategy}: missing/duplicate sample IDs")
            sizes[(record["dataset"], strategy)] = requested
    if not sizes:
        raise ValueError("no annotated target-alternative sets found")
    return sizes


def nested_sample_sizes(sizes: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for (dataset, strategy), size in sorted(sizes.items()):
        output[dataset][strategy] = size
    return dict(output)


def iter_comparisons(
    inputs: Sequence[str | Path],
    dataset_filter: str,
    sample_sizes: dict[tuple[str, str], int],
) -> Iterable[Comparison]:
    """Use the already selected pool without re-ranking or re-filtering."""

    for record in iter_item_records(inputs, dataset_filter):
        groups = alternatives_by_strategy(record["alternatives"])
        for strategy in sorted(groups):
            if (record["dataset"], strategy) not in sample_sizes:
                continue
            alternatives = groups[strategy]
            for alternative in alternatives:
                yield Comparison(
                    dataset=record["dataset"],
                    document_id=record["document_id"],
                    item_id=record["item_id"],
                    sentence_index=record["sentence_index"],
                    language=record["language"],
                    strategy=strategy,
                    alternative_id=str(alternative.get("id") or ""),
                    sample_index=alternative.get("sample_index", ""),
                    context_sentences=record["context_sentences"],
                    target=record["target"],
                    alternative=alternative,
                )


def generalised_jaccard(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    if denominator == 0:
        return 0.0
    numerator = sum(min(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in keys)
    return 1.0 - numerator / denominator


def context_activation(context: BranchFeatures, sentence_count: int, decay: float, relevant=None) -> dict[int, float]:
    by_sentence: dict[int, set[int]] = defaultdict(set)
    for mention in context.mentions:
        if relevant is not None and mention.cluster_id not in relevant:
            continue
        by_sentence[mention.sentence_index].add(mention.cluster_id)
    activation: dict[int, float] = defaultdict(float)
    for sentence_index, entity_ids in by_sentence.items():
        weight = decay ** (sentence_count - sentence_index - 1)
        for entity_id in entity_ids:
            activation[entity_id] += weight
    return dict(activation)


def recurrence_distances(
    context: BranchFeatures,
    target: BranchFeatures,
    alternative: BranchFeatures,
    sentence_count: int,
    decay: float,
) -> tuple[float, float]:
    context_ids = set(context.clusters)
    target_ids = set(target.clusters) & context_ids
    alternative_ids = set(alternative.clusters) & context_ids
    uniform_target = {entity_id: 1.0 for entity_id in target_ids}
    uniform_alternative = {entity_id: 1.0 for entity_id in alternative_ids}
    # Rescale all relevant activation weights by one common factor. This leaves
    # Soergel/Jaccard unchanged and prevents all old entities underflowing to 0.
    relevant = target_ids | alternative_ids
    latest = max((m.sentence_index for m in context.mentions if m.cluster_id in relevant), default=sentence_count - 1)
    activation = context_activation(context, latest + 1, decay, relevant)
    exponential_target = {entity_id: activation[entity_id] for entity_id in target_ids}
    exponential_alternative = {entity_id: activation[entity_id] for entity_id in alternative_ids}
    return (
        generalised_jaccard(uniform_target, uniform_alternative),
        generalised_jaccard(exponential_target, exponential_alternative),
    )


def new_head_counts(branch: BranchFeatures, context_ids: set[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entity_id, mentions in branch.clusters.items():
        if entity_id not in context_ids:
            counts[mentions[0].head_surface] += 1
    return counts


def new_head_distance(context: BranchFeatures, target: BranchFeatures, alternative: BranchFeatures) -> float:
    context_ids = set(context.clusters)
    return generalised_jaccard(
        new_head_counts(target, context_ids),
        new_head_counts(alternative, context_ids),
    )


def first_roles(branch: BranchFeatures) -> dict[int, str]:
    roles = {}
    for mention in branch.mentions:
        roles.setdefault(mention.cluster_id, mention.role)
    return roles


def last_roles(branch: BranchFeatures) -> dict[int, str]:
    roles = {}
    for mention in branch.mentions:
        roles[mention.cluster_id] = mention.role
    return roles


def transition_distance(
    context: BranchFeatures,
    target: BranchFeatures,
    alternative: BranchFeatures,
    transition_costs: dict[str, dict[str, float]],
) -> tuple[float, int]:
    """Compare costs over recurrent entities present in either continuation.

    An absent entity has value zero. The count is the size of the recurrent
    entity union; if it is empty, both representations have distance zero.
    """

    previous = last_roles(context)
    target_current = first_roles(target)
    alternative_current = first_roles(alternative)
    recurrent = sorted(set(previous) & (set(target_current) | set(alternative_current)))
    numerator = denominator = 0.0
    for entity_id in recurrent:
        costs = transition_costs[previous[entity_id]]
        target_value = (
            float(costs[target_current[entity_id]])
            if entity_id in target_current else 0.0
        )
        alternative_value = (
            float(costs[alternative_current[entity_id]])
            if entity_id in alternative_current else 0.0
        )
        numerator += abs(target_value - alternative_value)
        denominator += max(target_value, alternative_value)
    distance = numerator / denominator if denominator else 0.0
    return distance, len(recurrent)


def last_role_and_sentence(branch: BranchFeatures) -> dict[int, tuple[str, int]]:
    """Each entity's role at its most recent context mention, and which sentence."""
    seen = {}
    for mention in branch.mentions:
        seen[mention.cluster_id] = (mention.role, mention.sentence_index)
    return seen


def transition_distance_absent(
    context: BranchFeatures,
    target: BranchFeatures,
    alternative: BranchFeatures,
    model: dict,
    context_length: int,
) -> tuple[float, int]:
    """d3 with absence as a modelled state instead of a zero.

    The released d3 gives an absent entity the value 0, so an entity realized by
    one continuation and not the other contributes |c - 0| / max(c, 0) = 1
    whatever the entity was, and an entity absent from BOTH is dropped from the
    comparison altogether. Both are wrong for the same reason: 0 bits means a
    free transition, not a missing one.

    Here every entity established in the context takes part, and absence is
    scored by the GUM-trained model (entity_transition_model.py) as
    -log2 P(absent | previous role, sentences since the last mention). Two
    continuations that both ignore an entity then agree about it, which adds to
    the denominator and not the numerator. Because the model conditions on
    recency, a long-forgotten entity being absent from both costs almost nothing
    (about 0.01 bits at a gap of seven or more sentences), so stale entities
    dilute the distance only slightly and no arbitrary cut-off is needed.
    """
    spec = model["channels"]["role"]["model"]
    buckets = model["distance_buckets"]

    def bucket_of(distance):
        edges = (1, 2, 3, 6)
        for edge, label in zip(edges, buckets):
            if distance <= edge:
                return label
        return buckets[-1]

    previous = last_role_and_sentence(context)
    target_current = first_roles(target)
    alternative_current = first_roles(alternative)
    numerator = denominator = 0.0
    for entity_id, (role, sentence_index) in sorted(previous.items()):
        bucket = bucket_of(max(1, context_length - int(sentence_index)))
        costs = spec[role][bucket]["cost_bits"]
        target_value = float(costs[target_current.get(entity_id, "absent")])
        alternative_value = float(costs[alternative_current.get(entity_id, "absent")])
        numerator += abs(target_value - alternative_value)
        denominator += max(target_value, alternative_value)
    distance = numerator / denominator if denominator else 0.0
    return distance, len(previous)


def calculate_row(
    comparison: Comparison,
    costs: dict,
    decay: float,
    text_scorer: TextDistanceScorer,
    absent_model: dict | None = None,
) -> dict:
    context = branch_features(comparison.context_sentences)
    target = branch_features(branch_sentences(comparison.target))
    alternative = branch_features(branch_sentences(comparison.alternative))
    uniform, exponential = recurrence_distances(
        context, target, alternative, len(comparison.context_sentences), decay
    )
    transition, n_transition_entities = transition_distance(context, target, alternative, costs)
    row = {
        "dataset": comparison.dataset,
        "document_id": comparison.document_id,
        "item_id": comparison.item_id,
        "sentence_index": comparison.sentence_index,
        "language": comparison.language,
        "strategy": comparison.strategy,
        "alternative_id": comparison.alternative_id,
        "sample_index": comparison.sample_index,
        "n_context_entities": len(context.clusters),
        "n_target_entities": len(target.clusters),
        "n_alternative_entities": len(alternative.clusters),
        "n_transition_entities": n_transition_entities,
        "d1_recurrence_uniform": uniform,
        "d1_recurrence_exponential": exponential,
        "d2_new_head": new_head_distance(context, target, alternative),
        "d3_transition": transition,
    }
    if absent_model is not None:
        absent, n_absent = transition_distance_absent(
            context, target, alternative, absent_model, len(comparison.context_sentences))
        row["d3_transition_absent"] = absent
        row["n_context_entities_scored"] = n_absent
    row.update(
        text_scorer.distances(
            branch_text(comparison.target),
            branch_text(comparison.alternative),
        )
    )
    row.update(dependency_distances(comparison.target, comparison.alternative))
    for metric in (*METRICS, *(("d3_transition_absent",) if absent_model is not None else ())):
        if not math.isfinite(row[metric]):
            raise ValueError(f"non-finite {metric}: {comparison.item_id}/{comparison.alternative_id}")
    return row


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarise_rows(rows: Sequence[dict], metrics: Sequence[str]) -> list[dict]:
    group_fields = (
        "dataset",
        "document_id",
        "item_id",
        "sentence_index",
        "language",
        "strategy",
    )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["n_alternatives"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"missing/non-finite {metric} in {key}")
            summary[f"{metric}_mean"] = sum(values) / len(values)
            summary[f"{metric}_p80"] = quantile(values, 0.8)
            summary[f"{metric}_min"] = min(values)
        output.append(summary)
    return output


def atomic_csv(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="annotated JSON files or directories")
    parser.add_argument("--transition-costs", type=Path, required=True)
    parser.add_argument("--absent-transition-model", type=Path,
                        help="entity_transition_model.json; adds d3_transition_absent, which "
                             "scores absence instead of dropping it or calling it 0 bits")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("auto", "naturalstories", "clasp"), default="auto")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--samples-per-strategy", type=int, default=100)
    parser.add_argument("--activation-decay", type=float, default=0.5)
    parser.add_argument("--role-inventory", choices=tuple(INVENTORIES), default="fine",
                        help="fine (default): 12 roles, separating possessors, passive "
                             "subjects, agents, nominal modifiers and predicates, and "
                             "inheriting through appos as well as conj; coarse: the "
                             "original four states")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--semantic-model", default="sentence-transformers/all-distilroberta-v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--spacy-batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 0 < args.activation_decay <= 1:
        parser.error("--activation-decay must be in (0, 1]")
    if min(args.samples_per_strategy, args.spacy_batch_size, args.embedding_batch_size) <= 0:
        parser.error("sample and batch sizes must be positive")
    if len(set(args.strategies)) != len(args.strategies) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in args.strategies):
        parser.error("strategies must be unique simple names")
    global ROLES, ROLE_INVENTORY
    ROLE_INVENTORY = args.role_inventory
    with args.transition_costs.open(encoding="utf-8") as stream:
        cost_file = json.load(stream)
    costs = cost_file["transition_costs"]
    # The deprel inventory has no fixed state list: it is whatever the cost
    # model was fitted on, so take it from the file and check it agrees.
    ROLES = tuple(cost_file.get("roles") or INVENTORIES[ROLE_INVENTORY])
    stored = cost_file.get("role_inventory")
    if stored and stored != ROLE_INVENTORY:
        parser.error(f"--role-inventory {ROLE_INVENTORY} but {args.transition_costs} "
                     f"was fitted with {stored}")
    if set(ROLES) != set(costs):
        parser.error(f"{args.transition_costs}: states {sorted(set(ROLES) ^ set(costs))} "
                     "do not match its own cost table")
    validate_transition_costs(costs)
    absent_model = None
    metrics = list(METRICS)
    if args.absent_transition_model:
        with args.absent_transition_model.open(encoding="utf-8") as stream:
            absent_model = json.load(stream)
        if "role" not in absent_model.get("channels", {}):
            parser.error("the transition model has no role channel")
        metrics.append("d3_transition_absent")
    sizes = choose_strategy_sample_sizes(args.inputs, args.dataset, args.samples_per_strategy, args.strategies)
    scorer = TextDistanceScorer(args.spacy_model, args.semantic_model, args.device,
                                args.spacy_batch_size, args.embedding_batch_size)
    summaries = []
    # Work one document at a time; release text/embedding caches between files.
    # CLASP's five targets reuse one embedded alternative pool.
    for path in iter_annotation_files(args.inputs):
        comparisons = list(iter_comparisons([path], args.dataset, sizes))
        if not comparisons:
            continue
        texts = [branch_text(branch) for pair in comparisons for branch in (pair.target, pair.alternative)]
        scorer.prepare(texts)
        rows = [calculate_row(pair, costs, args.activation_decay, scorer, absent_model)
                for pair in comparisons]
        summaries.extend(summarise_rows(rows, metrics))
        print(f"computed {path.name}: {len(rows)} pairs", flush=True)
    for (dataset, strategy), size in sorted(sizes.items()):
        selected = [row for row in summaries if row["dataset"] == dataset and row["strategy"] == strategy]
        if not selected or any(row["n_alternatives"] != size for row in selected):
            raise ValueError(f"unexpected final sample count for {dataset}/{strategy}")
        output = args.out_dir / dataset / f"iv_{strategy}.csv"
        atomic_csv(output, selected)
        print(f"wrote {output}: {len(selected)} items, N={size}", flush=True)
    config = {
        "schema_version": 7, "strategies": args.strategies,
        "role_inventory": ROLE_INVENTORY, "roles": list(ROLES),
        "samples_per_strategy": nested_sample_sizes(sizes),
        "activation_decay": args.activation_decay,
        "transition_cost_file": str(args.transition_costs),
        "transition_costs": costs,
        "spacy_model": args.spacy_model, "semantic_model": args.semantic_model,
        "semantic_device": scorer.device,
        "metrics": metrics, "aggregations": ["mean", "p80", "min"],
        "absent_transition_model": str(args.absent_transition_model or ""),
        "selection": "exactly N already annotated draws per item/strategy; duplicates preserved",
        "missing_values": "empty/empty representations give 0; corrupt annotations/non-finite values raise errors",
        "dependency_representation": "base UD DEPREL in token order; root/punct retained; ngrams do not cross sentences",
        "dependency_aliases": {"nsubjpass": "nsubj", "csubjpass": "csubj", "dobj": "obj", "auxpass": "aux"},
        "syntax_source": "supplied gold Natural Stories CSV; silver Stanza CLASP targets and all alternatives",
        "pos_overlap_source": "spaCy on target/alternative text, preserving the previous POS baseline",
        "pairwise_values_saved": False,
    }
    temporary = args.out_dir / "iv_config.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(args.out_dir / "iv_config.json")


if __name__ == "__main__":
    main()