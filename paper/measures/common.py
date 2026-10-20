"""Shared helpers for the per-utterance measures of the paper.

Every measure script reads the annotated alternatives (one JSON file per Natural Stories story or
CLASP context, written by scripts/entity_iv/annotate_{ns,clasp}.py), walks over the targets it
contains, and writes one CSV row per target and sampling strategy.

This module provides:
  * the record iterator over those JSON files (via scripts/entity_iv/calculate_entity_iv.py),
  * the extraction of mentions with their entity id, head lemma, form and grammatical role,
  * the representative head lemma of an entity (its first non-pronominal mention),
  * a cached sentence-transformers embedder for head lemmas,
  * small numeric helpers and the CSV writer.
"""

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "entity_iv"))

import calculate_entity_iv as entity_iv  # noqa: E402
from entity_roles import mention_head, mention_role  # noqa: E402
from form_inventory import mention_form  # noqa: E402

STRATEGIES = [
    "ancestral", "temp_075", "temp_125",
    "nucleus_08", "nucleus_085", "nucleus_09", "nucleus_095",
    "typical_02", "typical_03", "typical_085", "typical_095",
]

PRONOUN_FORMS = {"pron_personal", "pron_possessive", "pron_demonstrative", "pron_other", "pron_wh"}

EMBEDDING_MODEL = "all-distilroberta-v1"
EMBEDDING_DIM = 768


# ----------------------------------------------------------------------------- records

def iter_records(inputs):
    """Yield every target record (context sentences, target, alternatives) in the given JSON
    files or directories."""
    for path in entity_iv.iter_annotation_files(list(inputs)):
        for record in entity_iv.iter_item_records([path], "auto"):
            yield record
        print("scored", path.name, flush=True)


def target_sentences(record):
    return entity_iv.branch_sentences(record["target"])


def alternatives_by_strategy(record):
    """{strategy: [list of sentences of each alternative]} in a fixed strategy order."""
    groups = entity_iv.alternatives_by_strategy(record["alternatives"])
    return {
        strategy: [entity_iv.branch_sentences(alternative) for alternative in groups[strategy]]
        for strategy in sorted(groups)
    }


def record_key(record):
    return {
        "item_id": record["item_id"],
        "document_id": record["document_id"],
        "sentence_index": record["sentence_index"],
    }


# ----------------------------------------------------------------------------- mentions

GRAMMATICAL_FUNCTIONS = ("subject", "subject_passive", "object_direct", "object_indirect", "oblique", "other")
FUNCTION_RANK = {function: rank for rank, function in enumerate(GRAMMATICAL_FUNCTIONS)}   # lower = more prominent


def grammatical_function(deprel):
    """Six-way grammatical function of a mention from the dependency label of its head."""
    label = deprel.lower()
    if label in ("nsubj:pass", "csubj:pass"):
        return "subject_passive"
    if label.startswith("nsubj") or label.startswith("csubj"):
        return "subject"
    if label == "obj":
        return "object_direct"
    if label == "iobj":
        return "object_indirect"
    if label.startswith("obl"):
        return "oblique"
    return "other"


def information_status(annotation):
    """GUM information status -> OLD / MED / NEW (given / accessible / new)."""
    if annotation.startswith("giv"):
        return "OLD"
    if annotation.startswith("acc"):
        return "MED"
    return "NEW"


def agreement_features(token):
    """Number and Gender of a token from its UD feature string."""
    features = {}
    for pair in (token.get("feats") or "").split("|"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            if key in ("Number", "Gender"):
                features[key] = value
    return features


def agreement_compatible(a, b):
    return all(a[key] == b[key] for key in ("Number", "Gender") if key in a and key in b)


@dataclass
class Mention:
    entity: int
    lemma: str            # head lemma, lower-cased
    form: str             # referring form (form_inventory.mention_form)
    is_subject: bool      # head is nsubj / csubj (active or passive)
    sentence: int         # index of the sentence within the utterance / context
    token_start: int
    token_end: int
    head: int             # token index of the head within the sentence
    function: str = "other"        # grammatical_function of the head's dependency label
    upos: str = ""                 # part of speech of the head
    agreement: dict = None         # Number / Gender of the head


def mention_order(mention):
    return (int(mention["token_start"]), -int(mention["token_end"]), int(mention["cluster_id"]))


def mentions_of(sentences):
    """All mentions of a list of sentences, in reading order."""
    out = []
    for sentence_index, sentence in enumerate(sentences):
        tokens = sentence["tokens"]
        for mention in sorted(sentence.get("mentions", []), key=mention_order):
            start = int(mention["token_start"])
            end = int(mention["token_end"])
            head = mention_head(tokens, start, end)
            deprel = mention_role(tokens, head, "deprel").lower()
            out.append(Mention(
                entity=int(mention["cluster_id"]),
                lemma=(tokens[head].get("lemma") or tokens[head]["text"]).lower(),
                form=mention_form(tokens, start, end),
                is_subject=deprel.startswith("nsubj") or deprel.startswith("csubj"),
                sentence=sentence_index,
                token_start=start,
                token_end=end,
                head=head,
                function=grammatical_function(deprel),
                upos=tokens[head].get("upos", ""),
                agreement=agreement_features(tokens[head]),
            ))
    return out


def per_entity(mentions):
    """entity -> {function: its most prominent function in the utterance, n: mentions, form: form of its
    first mention} (the summary the next-mention and referring-form measures work with)."""
    summary = {}
    for mention in mentions:
        entry = summary.setdefault(mention.entity, {"function": mention.function, "n": 0, "form": mention.form})
        entry["n"] += 1
        if FUNCTION_RANK[mention.function] < FUNCTION_RANK[entry["function"]]:
            entry["function"] = mention.function
    return summary


# ----------------------------------------------------------------------------- information status

GOLD_ENTITY = re.compile(r"\((\d+)-([a-z]+)-([a-z:]+)-")


def load_gold_status(csv_path):
    """{(sentence id, entity): GUM information status} from a manual story annotation CSV
    (semicolon-separated, `sent_id` 0-based, entity brackets in `entity_layer`)."""
    status = {}
    with open(csv_path) as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            for match in GOLD_ENTITY.finditer(row["entity_layer"]):
                status[(int(row["sent_id"]), int(match.group(1)))] = match.group(3)
    return status


def context_status(record, gold_dir=None):
    """{(absolute sentence id, entity): status annotation} of the context: from the manual annotation
    on Natural Stories (gold_dir/story_<k>_entity_annotated.csv), from CorPipe's information status
    on CLASP (given -> giv, else new)."""
    if record["dataset"] == "naturalstories":
        story = int(record["document_id"].split("_")[1])
        return load_gold_status(Path(gold_dir) / f"story_{story}_entity_annotated.csv")
    status = {}
    n_context = len(record["context_sentences"])
    for offset, sentence in enumerate(record["context_sentences"]):
        sentence_id = record["sentence_index"] - n_context + offset
        for mention in sentence.get("mentions", []):
            key = (sentence_id, int(mention["cluster_id"]))
            status.setdefault(key, "giv" if mention.get("information_status") == "given" else "new")
    return status


def entities_of(mentions):
    """Entity ids in order of first mention."""
    seen = []
    for mention in mentions:
        if mention.entity not in seen:
            seen.append(mention.entity)
    return seen


def representative_lemmas(mentions):
    """entity -> head lemma of its first non-pronominal mention (fallback: its first mention)."""
    first_any = {}
    first_nominal = {}
    for mention in mentions:
        first_any.setdefault(mention.entity, mention.lemma)
        if mention.form not in PRONOUN_FORMS:
            first_nominal.setdefault(mention.entity, mention.lemma)
    return {entity: first_nominal.get(entity, first_any[entity]) for entity in first_any}


# ----------------------------------------------------------------------------- embeddings

class Embedder:
    """Unit-norm sentence-transformers embeddings of single words, cached per word."""

    def __init__(self, model_name=EMBEDDING_MODEL):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = {}

    def vector(self, word):
        if word not in self.cache:
            encoded = self.model.encode([word], normalize_embeddings=True, show_progress_bar=False)
            self.cache[word] = encoded[0].astype(np.float64)
        return self.cache[word]

    def sentence_vectors(self, texts):
        """Unit-norm embeddings of whole sentences (for the semantic non-entity measure)."""
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


# ----------------------------------------------------------------------------- numerics

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def mean_over_pairs(items, distance):
    """Mean of distance(a, b) over all ordered pairs of distinct items (NaN for fewer than two)."""
    k = len(items)
    if k < 2:
        return float("nan")
    total = sum(distance(items[i], items[j]) for i in range(k) for j in range(k) if i != j)
    return total / (k * (k - 1))


def write_csv(rows, path):
    """Write dict rows to CSV; the header is the union of keys in order of first appearance."""
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path, len(rows))
