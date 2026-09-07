#!/usr/bin/env python3
"""Estimate last-mention role-transition costs from GUM coreference chains.

For each entity chain, every mention is paired with that entity's immediately
preceding mention in the document, even when intervening sentences do not
mention the entity.  Costs are negative log2 conditional probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROLES = ("subject", "object", "oblique", "other")
OPEN_RE = re.compile(r"\((\d+)-")
CLOSE_RE = re.compile(r"(\d+)\)")
ENTITY_RE = re.compile(r"(?:^|\|)Entity=([^|]+)")


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _close(
    stack: list[tuple[int, int]],
    spans: list[tuple[int, int, int]],
    token_index: int,
    expected_id: int | None = None,
) -> None:
    if not stack:
        raise ValueError(f"closing unopened entity at token {token_index}")
    if expected_id is None:
        entity_id, start = stack.pop()
    else:
        matches = [
            i for i, (entity_id, _start) in enumerate(stack)
            if entity_id == expected_id
        ]
        if not matches:
            raise ValueError(f"closing unopened entity {expected_id} at token {token_index}")
        entity_id, start = stack.pop(matches[-1])
    spans.append((start, token_index, entity_id))


def entity_spans(tokens: Sequence[dict]) -> list[tuple[int, int, int]]:
    """Parse the nested/overlapping GUM Entity value in CoNLL-U MISC."""

    stack: list[tuple[int, int]] = []
    spans: list[tuple[int, int, int]] = []
    for token_index, token in enumerate(tokens):
        match = ENTITY_RE.search(token["misc"])
        if not match:
            continue
        layer = match.group(1)
        position = 0
        while position < len(layer):
            opened = OPEN_RE.match(layer, position)
            if opened:
                stack.append((int(opened.group(1)), token_index))
                position = opened.end()
                continue
            # An explicit close is either the first event or follows another
            # close.  This avoids treating numeric metadata as an entity ID.
            closed = (
                CLOSE_RE.match(layer, position)
                if position == 0 or layer[position - 1] == ")"
                else None
            )
            if closed:
                _close(stack, spans, token_index, int(closed.group(1)))
                position = closed.end()
                continue
            if layer[position] == ")":
                _close(stack, spans, token_index)
            position += 1
    if stack:
        raise ValueError(f"unclosed entity IDs: {[entity_id for entity_id, _ in stack]}")
    return sorted(spans, key=lambda span: (span[0], -span[1], span[2]))


def parse_conllu(text: str) -> list[list[dict]]:
    sentences, current = [], []
    for raw_line in text.splitlines() + [""]:
        line = raw_line.rstrip("\r\n")
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 10:
            raise ValueError(f"expected 10 CoNLL-U columns, got {len(columns)}")
        if "-" in columns[0] or "." in columns[0]:
            continue
        current.append({
            "id": int(columns[0]),
            "text": columns[1],
            "lemma": columns[2],
            "upos": columns[3],
            "xpos": columns[4],
            "head": int(columns[6]),
            "deprel": columns[7],
            "misc": columns[9],
        })
    return sentences


def iter_gum_documents(path: str | Path) -> Iterable[tuple[str, str]]:
    """Yield released GUM dep/*.conllu files from a directory or zip."""

    source = Path(path)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = sorted(
                name for name in archive.namelist()
                if re.search(r"(?:^|/)dep/GUM_[^/]+\.conllu$", name)
                and "/_build/" not in name
            )
            for name in names:
                yield Path(name).name, archive.read(name).decode("utf-8-sig")
        return
    if not source.exists():
        raise FileNotFoundError(source)
    files = sorted(
        file for file in source.rglob("GUM_*.conllu")
        if "_build" not in file.parts and "dep" in file.parts
    )
    if source.is_dir() and source.name == "dep":
        files = sorted(source.glob("GUM_*.conllu"))
    for file in files:
        yield file.name, file.read_text(encoding="utf-8-sig")


def mention_head(tokens: Sequence[dict], start: int, end: int) -> int:
    span_ids = {int(tokens[index]["id"]) for index in range(start, end + 1)}
    candidates = [
        index for index in range(start, end + 1)
        if int(tokens[index]["head"]) not in span_ids
    ]
    if not candidates:
        candidates = list(range(start, end + 1))
    content = [
        index for index in candidates
        if tokens[index]["upos"] != "PUNCT"
    ]
    return (content or candidates)[0]


def inherited_relation(tokens: Sequence[dict], head_index: int) -> str:
    """A conjunct inherits its governor's grammatical relation."""

    seen = set()
    index = head_index
    while index not in seen:
        seen.add(index)
        relation = str(tokens[index]["deprel"]).lower().split(":", 1)[0]
        if relation != "conj":
            return relation
        governor = int(tokens[index]["head"])
        by_id = {int(token["id"]): i for i, token in enumerate(tokens)}
        if governor not in by_id:
            return relation
        index = by_id[governor]
    return "other"


def role_of_mention(tokens: Sequence[dict], start: int, end: int) -> str:
    relation = inherited_relation(tokens, mention_head(tokens, start, end))
    if relation in {"nsubj", "csubj"}:
        return "subject"
    if relation in {"obj", "dobj", "iobj", "ccomp", "xcomp"}:
        return "object"
    if relation in {"obl", "nmod"}:
        return "oblique"
    return "other"


def document_mentions(text: str) -> list[tuple[int, int, int, int, str]]:
    mentions = []
    for sentence_index, tokens in enumerate(parse_conllu(text)):
        for start, end, entity_id in entity_spans(tokens):
            mentions.append((sentence_index, start, end, entity_id, role_of_mention(tokens, start, end)))
    return sorted(mentions, key=lambda item: (item[0], item[1], -item[2], item[3]))


def estimate(path: str | Path, smoothing: float) -> dict:
    counts: Counter[tuple[str, str]] = Counter()
    document_count = mention_count = transition_count = chain_count = 0
    for name, text in iter_gum_documents(path):
        document_count += 1
        last_role: dict[int, str] = {}
        seen_entities = set()
        mentions = document_mentions(text)
        mention_count += len(mentions)
        for _sent, _start, _end, entity_id, role in mentions:
            seen_entities.add(entity_id)
            if entity_id in last_role:
                counts[(last_role[entity_id], role)] += 1
                transition_count += 1
            last_role[entity_id] = role
        chain_count += len(seen_entities)
        print(f"{name}: mentions={len(mentions)}, chains={len(seen_entities)}", flush=True)

    if not document_count:
        raise ValueError(f"no released GUM dep/GUM_*.conllu files found under {path}")

    probabilities: dict[str, dict[str, float]] = defaultdict(dict)
    costs: dict[str, dict[str, float]] = defaultdict(dict)
    nested_counts: dict[str, dict[str, int]] = defaultdict(dict)
    row_totals = {}
    for previous in ROLES:
        row_total = sum(counts[(previous, current)] for current in ROLES)
        row_totals[previous] = row_total
        denominator = row_total + smoothing * len(ROLES)
        for current in ROLES:
            probability = (counts[(previous, current)] + smoothing) / denominator
            nested_counts[previous][current] = counts[(previous, current)]
            probabilities[previous][current] = probability
            costs[previous][current] = -math.log2(probability)

    return {
        "schema_version": 1,
        "source": "GUM released dep/*.conllu Entity annotations",
        "roles": list(ROLES),
        "smoothing": smoothing,
        "cost_unit": "bits",
        "document_count": document_count,
        "entity_chain_count": chain_count,
        "mention_count": mention_count,
        "transition_count": transition_count,
        "row_totals": row_totals,
        "counts": dict(nested_counts),
        "probabilities": dict(probabilities),
        "transition_costs": dict(costs),
        "semantics": "previous state is the same entity's last mention anywhere in prior document context",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gum", required=True, help="GUM directory or release zip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoothing", type=float, default=0.5)
    args = parser.parse_args()
    if args.smoothing <= 0:
        parser.error("--smoothing must be positive")
    output = estimate(args.gum, args.smoothing)
    atomic_json(args.output, output)
    print(f"wrote {args.output}; transitions={output['transition_count']}", flush=True)


if __name__ == "__main__":
    main()
