#!/usr/bin/env python3
"""Calculate entity and text IV distances for temp_125 alternatives only.

Distance 1 compares recurrence of context entities (uniform and exponentially
decayed activation).  Distance 2 compares multisets of newly introduced entity
heads.  Distance 3 compares GUM-estimated role-transition costs from each
entity's last context mention to its first mention in the target/alternative.
Lexical, syntactic, and semantic distances follow the implementation in
dmg-illc/information-value.
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
    "d_syntactic_1gram",
    "d_syntactic_2gram",
    "d_syntactic_3gram",
    "d_semantic_cosine",
    "d_semantic_euclidean",
)
HTML_TAG = re.compile(r"<[^>]*>")
SENTENCE_ENDINGS = (".", "!", "?", "…", "。", "！", "？")
TRAILING_CLOSERS = "\"'”’»)]}"
SELECTED_STRATEGY = "temp_125"


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
    form: str


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


def is_complete_sentence(text: str) -> bool:
    """Use terminal punctuation as a transparent completeness heuristic."""

    plain = HTML_TAG.sub("", text or "").strip()
    plain = plain.rstrip(TRAILING_CLOSERS).rstrip()
    return plain.endswith(SENTENCE_ENDINGS)


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
        self.syntactic: dict[str, list[str]] = {}
        self.embeddings: dict[str, object] = {}

    @staticmethod
    def make_ngrams(values: Sequence[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(values[index:index + n]) for index in range(len(values) - n + 1)]

    @staticmethod
    def inverse_overlap(
        left: Sequence[tuple[str, ...]],
        right: Sequence[tuple[str, ...]],
    ) -> float | None:
        """Implement scorer.py's symmetric inverse-overlap distance."""

        denominator = len(left) + len(right)
        if denominator == 0:
            return None
        left_types, right_types = set(left), set(right)
        shared = sum(value in left_types for value in right)
        shared += sum(value in right_types for value in left)
        return (denominator - shared) / denominator

    def prepare(self, texts: Sequence[str]) -> None:
        """Parse and embed every unique target/alternative text in batches."""

        unique = list(dict.fromkeys(texts))
        print(f"preparing text distances for {len(unique)} unique utterances", flush=True)
        for text, doc in zip(
            unique,
            self.nlp.pipe(unique, batch_size=self.spacy_batch_size),
        ):
            self.lexical[text] = [token.text.lower() for token in doc]
            self.syntactic[text] = [token.pos_ for token in doc]

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
                self.embeddings[text] = embedding
            completed = min(start + len(batch), len(unique))
            batch_number = start // self.embedding_batch_size + 1
            if completed == len(unique) or batch_number % 100 == 0:
                print(f"embedded {completed}/{len(unique)} utterances", flush=True)

    def distances(self, target: str, alternative: str) -> dict[str, float | None]:
        output: dict[str, float | None] = {}
        for n in range(1, 4):
            target_words = self.make_ngrams(self.lexical[target], n)
            alternative_words = self.make_ngrams(self.lexical[alternative], n)
            output[f"d_lexical_{n}gram"] = self.inverse_overlap(
                target_words, alternative_words
            )

            target_pos = self.make_ngrams(self.syntactic[target], n)
            alternative_pos = self.make_ngrams(self.syntactic[alternative], n)
            output[f"d_syntactic_{n}gram"] = self.inverse_overlap(
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


def mention_head(tokens: Sequence[dict], start: int, end: int) -> int:
    token_ids = {int(tokens[index].get("id", index + 1)) for index in range(start, end + 1)}
    candidates = [
        index for index in range(start, end + 1)
        if int(tokens[index].get("head", 0)) not in token_ids
    ]
    if not candidates:
        candidates = list(range(start, end + 1))
    content = [
        index for index in candidates
        if str(tokens[index].get("upos", "")).upper() != "PUNCT"
        and str(tokens[index].get("xpos", "")) not in {".", ",", ":", "-LRB-", "-RRB-"}
    ]
    return (content or candidates)[0]


def inherited_relation(tokens: Sequence[dict], head_index: int) -> str:
    by_id = {int(token.get("id", index + 1)): index for index, token in enumerate(tokens)}
    seen = set()
    index = head_index
    while index not in seen:
        seen.add(index)
        relation = relation_base(str(tokens[index].get("deprel", "_")))
        if relation != "conj":
            return relation
        governor = int(tokens[index].get("head", 0))
        if governor not in by_id:
            break
        index = by_id[governor]
    return "other"


def mention_role(tokens: Sequence[dict], head_index: int) -> str:
    relation = inherited_relation(tokens, head_index)
    if relation in {"nsubj", "csubj"}:
        return "subject"
    if relation in {"obj", "iobj", "ccomp", "xcomp"}:
        return "object"
    if relation in {"obl", "nmod"}:
        return "oblique"
    return "other"


def _pos(token: dict) -> tuple[str, str]:
    return str(token.get("upos", "")).upper(), str(token.get("xpos", "")).upper()


def mention_form(tokens: Sequence[dict], start: int, end: int, head_index: int) -> str:
    """Return an English referring-form class from the Givenness Hierarchy.

    These are observable forms, not inferred cognitive statuses.  In
    particular, syntax alone cannot distinguish demonstrative ``this N`` from
    indefinite ``this N``, so proximal demonstrative NPs remain one class.
    """

    head = tokens[head_index]
    upos, xpos = _pos(head)
    head_word = normalised_surface(head.get("text", ""))
    demonstratives = {"this", "that", "these", "those"}
    if head_word in demonstratives and (upos == "PRON" or xpos in {"DT", "PDT"}):
        return "demonstrative_pronoun"
    if xpos in {"WP", "WP$"}:
        return "other"
    if upos == "PRON" or xpos in {"PRP", "PRP$"}:
        return "personal_pronoun"

    head_id = int(head.get("id", head_index + 1))
    modifiers = [
        token for token in tokens[start:end + 1]
        if int(token.get("head", 0)) == head_id
    ]
    words = {normalised_surface(token.get("text", "")) for token in modifiers}
    if words & {"this", "these"}:
        return "proximal_demonstrative_np"
    if words & {"that", "those"}:
        return "distal_demonstrative_np"
    if "the" in words:
        return "definite_np"
    if words & {"a", "an"}:
        return "indefinite_np"
    return "other"


def branch_features(sentences: Sequence[dict]) -> BranchFeatures:
    features = []
    ordinal = 0
    for sentence_index, sentence in enumerate(sentences):
        tokens = sentence.get("tokens", [])
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
                role=mention_role(tokens, head_index),
                form=mention_form(tokens, start, end, head_index),
            )
            features.append(feature)
            ordinal += 1
    clusters: dict[int, list[MentionFeature]] = defaultdict(list)
    for mention in features:
        clusters[mention.cluster_id].append(mention)
    return BranchFeatures(features, dict(clusters))


def branch_sentences(branch: dict) -> list[dict]:
    if not isinstance(branch, dict) or not isinstance(branch.get("sentences"), list):
        raise ValueError("branch must contain a sentences list")
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
        try:
            with path.open(encoding="utf-8") as stream:
                document = json.load(stream)
        except (json.JSONDecodeError, OSError):
            continue
        dataset = document.get("dataset")
        if dataset not in {"naturalstories", "clasp"}:
            continue
        if dataset_filter != "auto" and dataset != dataset_filter:
            continue
        full_context = document.get("context", [])
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
                continue
            language = str(item.get("language") or target.get("language") or "")
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
    """Group successfully annotated alternatives by generation strategy."""

    groups: dict[str, list[dict]] = defaultdict(list)
    for alternative in alternatives:
        if (
            isinstance(alternative, dict)
            and isinstance(alternative.get("sentences"), list)
            and alternative["sentences"]
        ):
            strategy = str(alternative.get("strategy") or "unknown")
            if strategy == SELECTED_STRATEGY:
                groups[strategy].append(alternative)
    return dict(groups)


def choose_strategy_sample_sizes(
    inputs: Sequence[str | Path],
    dataset_filter: str,
    requested: int | None,
) -> dict[tuple[str, str], int]:
    """Choose a constant N across items separately for each strategy."""

    units: dict[str, list[dict[str, int]]] = defaultdict(list)
    strategies: dict[str, set[str]] = defaultdict(set)
    for record in iter_item_records(inputs, dataset_filter):
        counts = {
            strategy: len(alternatives)
            for strategy, alternatives in alternatives_by_strategy(
                record["alternatives"]
            ).items()
        }
        units[record["dataset"]].append(counts)
        strategies[record["dataset"]].update(counts)

    if not units:
        raise ValueError("no Natural Stories/CLASP target-alternative sets found")

    sizes = {}
    for dataset in sorted(units):
        for strategy in sorted(strategies[dataset]):
            minimum = min(counts.get(strategy, 0) for counts in units[dataset])
            if minimum == 0:
                raise ValueError(
                    f"dataset={dataset}, strategy={strategy} is missing from at least one item"
                )
            if requested is not None and requested > minimum:
                raise ValueError(
                    f"dataset={dataset}, strategy={strategy}: requested {requested} "
                    f"alternatives per item, but the minimum available is {minimum}"
                )
            sizes[(dataset, strategy)] = requested or minimum
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
    """Select complete sentences first within each strategy and yield pairs."""

    for record in iter_item_records(inputs, dataset_filter):
        groups = alternatives_by_strategy(record["alternatives"])
        for strategy in sorted(groups):
            limit = sample_sizes[(record["dataset"], strategy)]
            alternatives = groups[strategy]
            complete = [
                alternative for alternative in alternatives
                if is_complete_sentence(str(alternative.get("text", "")))
            ]
            incomplete = [
                alternative for alternative in alternatives
                if not is_complete_sentence(str(alternative.get("text", "")))
            ]
            for alternative in (complete + incomplete)[:limit]:
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


def context_activation(context: BranchFeatures, sentence_count: int, decay: float) -> dict[int, float]:
    by_sentence: dict[int, set[int]] = defaultdict(set)
    for mention in context.mentions:
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
    activation = context_activation(context, sentence_count, decay)
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
) -> tuple[float | None, int]:
    previous = last_roles(context)
    target_current = first_roles(target)
    alternative_current = first_roles(alternative)
    comparable = sorted(set(previous) & set(target_current) & set(alternative_current))
    if not comparable:
        return None, 0
    target_values, alternative_values = [], []
    for entity_id in comparable:
        previous_role = previous[entity_id]
        target_values.append(float(transition_costs[previous_role][target_current[entity_id]]))
        alternative_values.append(float(transition_costs[previous_role][alternative_current[entity_id]]))
    denominator = sum(max(left, right) for left, right in zip(target_values, alternative_values))
    distance = 0.0 if denominator == 0 else sum(
        abs(left - right) for left, right in zip(target_values, alternative_values)
    ) / denominator
    return distance, len(comparable)


def calculate_row(
    comparison: Comparison,
    costs: dict,
    decay: float,
    text_scorer: TextDistanceScorer,
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
        "alternative_is_complete": int(
            is_complete_sentence(str(comparison.alternative.get("text", "")))
        ),
        "n_context_entities": len(context.clusters),
        "n_target_entities": len(target.clusters),
        "n_alternative_entities": len(alternative.clusters),
        "n_transition_entities": n_transition_entities,
        "d1_recurrence_uniform": uniform,
        "d1_recurrence_exponential": exponential,
        "d2_new_head": new_head_distance(context, target, alternative),
        "d3_transition": transition,
    }
    row.update(
        text_scorer.distances(
            branch_text(comparison.target),
            branch_text(comparison.alternative),
        )
    )
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
        summary["n_complete_alternatives"] = sum(
            int(row.get("alternative_is_complete", 0)) for row in group
        )
        summary["n_incomplete_fallback_alternatives"] = (
            len(group) - summary["n_complete_alternatives"]
        )
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            summary[f"{metric}_n_valid"] = len(values)
            summary[f"{metric}_mean"] = sum(values) / len(values) if values else None
            summary[f"{metric}_p80"] = quantile(values, 0.8) if values else None
            summary[f"{metric}_min"] = min(values) if values else None
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="annotated JSON files or directories")
    parser.add_argument("--transition-costs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", choices=("auto", "naturalstories", "clasp"), default="auto")
    parser.add_argument("--activation-decay", type=float, default=0.5)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-distilroberta-v1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--spacy-batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--samples-per-strategy",
        type=int,
        help="fixed N for every item and strategy; omit to use each dataset/strategy minimum",
    )
    args = parser.parse_args()
    if not 0 < args.activation_decay <= 1:
        parser.error("--activation-decay must be in (0, 1]")
    if args.samples_per_strategy is not None and args.samples_per_strategy <= 0:
        parser.error("--samples-per-strategy must be positive")
    if args.spacy_batch_size <= 0 or args.embedding_batch_size <= 0:
        parser.error("batch sizes must be positive")

    with Path(args.transition_costs).open(encoding="utf-8") as stream:
        transition_model = json.load(stream)
    costs = transition_model["transition_costs"]
    sample_sizes = choose_strategy_sample_sizes(
        args.inputs, args.dataset, args.samples_per_strategy
    )
    for (dataset, strategy), size in sorted(sample_sizes.items()):
        print(f"{dataset}: {strategy}: N={size}", flush=True)
    comparisons = list(iter_comparisons(args.inputs, args.dataset, sample_sizes))
    if not comparisons:
        raise ValueError("no Natural Stories/CLASP target-alternative comparisons found")
    text_scorer = TextDistanceScorer(
        args.spacy_model,
        args.semantic_model,
        args.device,
        args.spacy_batch_size,
        args.embedding_batch_size,
    )
    texts = []
    for comparison in comparisons:
        texts.append(branch_text(comparison.target))
        texts.append(branch_text(comparison.alternative))
    text_scorer.prepare(texts)
    rows = [
        calculate_row(comparison, costs, args.activation_decay, text_scorer)
        for comparison in comparisons
    ]
    summaries = summarise_rows(rows, METRICS)
    output_dir = Path(args.out_dir)
    written = []
    for dataset in ("clasp", "naturalstories"):
        dataset_summaries = [
            summary for summary in summaries
            if summary["dataset"] == dataset
        ]
        if dataset_summaries:
            output_path = output_dir / dataset / "iv_temp_125.csv"
            atomic_csv(output_path, dataset_summaries)
            written.append((dataset, output_path, len(dataset_summaries)))
    config = {
        "schema_version": 4,
        "strategy": SELECTED_STRATEGY,
        "dataset_filter": args.dataset,
        "requested_samples_per_strategy": args.samples_per_strategy,
        "samples_per_strategy": nested_sample_sizes(sample_sizes),
        "activation_decay": args.activation_decay,
        "transition_cost_file": str(args.transition_costs),
        "spacy_model": args.spacy_model,
        "semantic_model": args.semantic_model,
        "semantic_device": text_scorer.device,
        "reference_implementation": "https://github.com/dmg-illc/information-value/tree/main/code",
        "new_entity_head": "NFKC-casefolded surface head on both target and alternative",
        "aggregation": "separate for every target item and generation strategy",
        "stored_values": "per-item mean, p80, and minimum; pairwise values are not written",
        "selection": "constant N within each dataset/strategy; complete sentences first and incomplete annotated alternatives only as fallback",
        "complete_sentence_test": "after removing HTML tags and trailing quote/bracket closers, text ends in . ! ? … 。！？",
        "distances": {
            "d1_recurrence_uniform": "generalised Jaccard on binary recurrent-context-entity presence",
            "d1_recurrence_exponential": "generalised Jaccard on recurrent presence weighted by decayed context activation",
            "d2_new_head": "generalised Jaccard on counts of new entities by normalized surface head",
            "d3_transition": "Soergel distance between GUM costs from each shared recurrent entity's last context role to its first current role",
            "d_lexical_1gram_to_3gram": "symmetric inverse overlap of lowercase spaCy token n-grams",
            "d_syntactic_1gram_to_3gram": "symmetric inverse overlap of spaCy POS n-grams",
            "d_semantic_cosine": "cosine distance between L2-normalized mean-pooled final-layer embeddings",
            "d_semantic_euclidean": "Euclidean distance between the same normalized embeddings",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "iv_config.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    temporary.replace(output_dir / "iv_config.json")
    for dataset, path, count in written:
        print(f"wrote {count} {dataset} item summaries to {path}", flush=True)


if __name__ == "__main__":
    main()
