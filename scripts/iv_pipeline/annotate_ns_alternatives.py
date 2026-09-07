#!/usr/bin/env python3
"""Annotate Natural Stories alternatives against manual CSV entity context.

The manual CSV is authoritative: its tokens, dependencies, and entity spans are
never retokenized.  Stanza parses generated alternatives only.  CorPipe then
links each alternative independently to the frozen manual prefix.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from corpipe26_seeded import SeededCorPipe
from dataset_adapters import StanzaSilverParser, validate_sentence_spec


SCHEMA_VERSION = 8
ANNOTATION_SCHEMA_VERSION = "entity_annotations_v4_native_tokens"
DEFAULT_MODEL = "ufal/corpipe26-twostage-corefud1.4-large-260702"
CSV_COLUMNS = [
    "story_id", "sent_id", "token_id", "deprel", "form", "head", "pos",
    "entity_layer",
]
OPEN_RE = re.compile(r"\((\d+)-")
CLOSE_RE = re.compile(r"(\d+)\)")
FRAGMENTED_MARKUP = "</em></strong><strong><em>"


def normalise_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def has_bad_unicode(text: str) -> bool:
    """Reject decoding damage before it reaches Stanza."""

    return "\ufffd" in text or any(0x80 <= ord(char) <= 0x9F for char in text)


def has_fragmented_character_markup(text: str) -> bool:
    """Detect formatting wrapped repeatedly around individual characters."""

    return text.lower().count(FRAGMENTED_MARKUP) >= 2


def load_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def story_number(path: str | Path) -> int:
    match = re.search(r"story_(\d+)_", Path(path).name)
    if not match:
        raise ValueError(f"cannot read story number from {path}")
    return int(match.group(1))


def _close_mention(
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
        candidates = [
            i for i, (entity_id, _start) in enumerate(stack)
            if entity_id == expected_id
        ]
        if not candidates:
            raise ValueError(
                f"closed unopened entity {expected_id} at token {token_index}"
            )
        entity_id, start = stack.pop(candidates[-1])
    spans.append((start, token_index, entity_id))


def entity_spans(rows: Sequence[dict[str, str]]) -> list[tuple[int, int, int]]:
    """Read nested/overlapping GUM-style spans from the CSV entity column."""

    stack: list[tuple[int, int]] = []
    spans: list[tuple[int, int, int]] = []
    for token_index, row in enumerate(rows):
        layer = row["entity_layer"]
        if not layer or layer == "_":
            continue
        layer = layer.removeprefix("Entity=")
        position = 0
        while position < len(layer):
            opened = OPEN_RE.match(layer, position)
            if opened:
                stack.append((int(opened.group(1)), token_index))
                position = opened.end()
                continue
            closed = (
                CLOSE_RE.match(layer, position)
                if position == 0 or layer[position - 1] == ")"
                else None
            )
            if closed:
                _close_mention(stack, spans, token_index, int(closed.group(1)))
                position = closed.end()
                continue
            if layer[position] == ")":
                _close_mention(stack, spans, token_index)
            position += 1
    if stack:
        raise ValueError(f"unclosed entities: {[entity_id for entity_id, _ in stack]}")
    return sorted(spans, key=lambda span: (span[0], -span[1], span[2]))


def raw_mention(
    sentence: dict, token_start: int, token_end: int, cluster_id: int
) -> dict:
    tokens = sentence["tokens"]
    char_start = int(tokens[token_start]["char_start"])
    char_end = int(tokens[token_end]["char_end"])
    return {
        "token_start": int(token_start),
        "token_end": int(token_end),
        "char_start": char_start,
        "char_end": char_end,
        "text": sentence["text"][char_start:char_end],
        "cluster_id": int(cluster_id),
    }


def add_token_entity_columns(sentence: dict) -> dict:
    output = copy.deepcopy(sentence)
    events: list[list[tuple[int, int, str]]] = [[] for _ in output["tokens"]]
    for mention in output.get("mentions", []):
        start = int(mention["token_start"])
        end = int(mention["token_end"])
        entity_id = int(mention["cluster_id"])
        if start == end:
            events[start].append((1, entity_id, f"({entity_id})"))
        else:
            events[start].append((0, -end, f"({entity_id}"))
            events[end].append((2, -start, f"{entity_id})"))
    for index, token in enumerate(output["tokens"]):
        token["form"] = token["text"]
        token["pos"] = token.get("xpos") or token.get("upos") or "X"
        values = [value for _kind, _order, value in sorted(events[index])]
        token["entity_layer"] = "|".join(values) if values else "_"
    return output


def _token_sort_key(token_id: str) -> tuple:
    parts = re.split(r"([0-9]+)", token_id)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def read_manual_story(path: str | Path, expected_story_id: int) -> list[dict]:
    """Construct the story directly from the annotated CSV token system."""

    rows_by_sentence: dict[int, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{path}: expected columns {CSV_COLUMNS}")
        for row in reader:
            if int(row["story_id"]) != expected_story_id:
                raise ValueError(f"{path}: mixed story IDs")
            rows_by_sentence[int(row["sent_id"])].append(row)

    sentence_ids = sorted(rows_by_sentence)
    if sentence_ids != list(range(len(sentence_ids))):
        raise ValueError(f"{path}: sentence IDs must be contiguous from zero")

    story = []
    for sent_id in sentence_ids:
        rows = sorted(rows_by_sentence[sent_id], key=lambda row: _token_sort_key(row["token_id"]))
        forms = [row["form"] for row in rows]
        text = " ".join(forms)
        tokens, cursor = [], 0
        for index, row in enumerate(rows, 1):
            start, end = cursor, cursor + len(row["form"])
            head = int(row["head"])
            if not 0 <= head <= len(rows):
                raise ValueError(
                    f"{path}: bad head {head} at sentence {sent_id}, token {index}"
                )
            tokens.append({
                "id": index,
                "source_token_id": row["token_id"],
                "text": row["form"],
                "lemma": "_",
                "upos": row["pos"],
                "xpos": row["pos"],
                "feats": "_",
                "head": head,
                "deprel": row["deprel"],
                "deps": "_",
                "misc": "_",
                "char_start": start,
                "char_end": end,
                "grammar_source": "naturalstories_manual_conllx_csv",
            })
            cursor = end + 1
        sentence = {
            "sentence_index": sent_id,
            "source_story_id": f"naturalstories_{expected_story_id}",
            "source_sentence_index": sent_id,
            "text": text,
            "tokens": tokens,
            "grammar_source": "naturalstories_manual_conllx_csv",
            "entity_source": "manual_gum_csv_native_tokens",
        }
        sentence["mentions"] = [
            raw_mention(sentence, start, end, entity_id)
            for start, end, entity_id in entity_spans(rows)
        ]
        validate_sentence_spec(sentence)
        story.append(add_token_entity_columns(sentence))
    return story


class CachedSilverParser:
    def __init__(self, parser: StanzaSilverParser):
        self.parser = parser
        self.cache: dict[str, dict] = {}

    def parse(self, text: str) -> dict:
        key = normalise_text(text)
        if not key:
            raise ValueError("cannot parse empty text")
        if key not in self.cache:
            parsed = self.parser.parse(key)
            for token in parsed["tokens"]:
                if token.get("text") == "<UNK>":
                    start = int(token["char_start"])
                    end = int(token["char_end"])
                    surface = parsed["text"][start:end]
                    if not surface:
                        raise ValueError("Stanza <UNK> token has an empty source span")
                    token["text"] = surface
                    if token.get("lemma") in {None, "", "_", "<UNK>"}:
                        token["lemma"] = surface
            validate_sentence_spec(parsed)
            self.cache[key] = parsed
        return copy.deepcopy(self.cache[key])


class StanzaSentenceSplitter:
    """Recover sentence boundaries, then let StanzaSilverParser parse each one."""

    def __init__(self, model_dir: str | None = None):
        try:
            import stanza  # type: ignore
        except ImportError as exc:
            raise RuntimeError("stanza is required") from exc
        options = {"lang": "en", "processors": "tokenize", "use_gpu": False, "verbose": False}
        if model_dir:
            options["dir"] = model_dir
        self.pipeline = stanza.Pipeline(**options)
        self.cache: dict[str, list[str]] = {}

    def split(self, text: str) -> list[str]:
        key = normalise_text(text)
        if not key:
            return []
        if key not in self.cache:
            document = self.pipeline(key)
            pieces = []
            for sentence in document.sentences:
                starts = [token.start_char for token in sentence.tokens if token.start_char is not None]
                ends = [token.end_char for token in sentence.tokens if token.end_char is not None]
                piece = key[min(starts):max(ends)] if starts and ends else sentence.text
                if normalise_text(piece):
                    pieces.append(normalise_text(piece))
            if not pieces:
                raise ValueError("Stanza returned no sentences")
            self.cache[key] = pieces
        return list(self.cache[key])


def parse_text(text: str, splitter: StanzaSentenceSplitter, silver: CachedSilverParser) -> list[dict]:
    return [silver.parse(sentence) for sentence in splitter.split(text)]


def clean_corpipe_sentence(sentence: dict) -> dict:
    clean = {
        "sentence_index": int(sentence.get("sentence_index", 0)),
        "text": sentence["text"],
        "grammar_source": sentence.get("grammar_source"),
        "tokens": [dict(token) for token in sentence["tokens"]],
    }
    clean["mentions"] = [
        raw_mention(
            clean,
            int(mention["token_start"]),
            int(mention["token_end"]),
            int(mention["cluster_id"]),
        )
        for mention in sentence.get("mentions", [])
    ]
    validate_sentence_spec(clean)
    return add_token_entity_columns(clean)


def load_alternatives(path: str | Path, limit: int) -> tuple[dict[int, list[dict]], dict]:
    alternatives: dict[int, list[dict]] = defaultdict(list)
    kept: Counter[tuple[int, str]] = Counter()
    examined: Counter[tuple[int, str]] = Counter()
    blank: Counter[tuple[int, str]] = Counter()
    bad_unicode: Counter[tuple[int, str]] = Counter()
    fragmented_markup: Counter[tuple[int, str]] = Counter()
    for row in load_jsonl(path):
        sent_id, strategy = int(row["sent_id"]), str(row["strategy"])
        key = sent_id, strategy
        for raw_value in row["raws"]:
            if kept[key] >= limit:
                break
            examined[key] += 1
            text = raw_value.get("text") if isinstance(raw_value, dict) else raw_value
            raw = raw_value.get("raw") if isinstance(raw_value, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"{path}: non-string alternative at {key}")
            if has_fragmented_character_markup(text):
                fragmented_markup[key] += 1
                continue
            if not text.strip():
                blank[key] += 1
                continue
            if has_bad_unicode(text):
                bad_unicode[key] += 1
                continue
            kept[key] += 1
            alternatives[sent_id].append({
                "id": f"{strategy}_{kept[key]}",
                "kind": "generated_alternative",
                "text": normalise_text(text),
                "raw": raw,
                "strategy": strategy,
                "sample_index": kept[key],
            })
    keys = sorted(set(examined) | set(kept))
    qa = {
        "source_file": str(path),
        "branch_limit_per_sentence_strategy": limit,
        "input_slots_examined": sum(examined.values()),
        "valid_alternatives_loaded": sum(kept.values()),
        "blank_alternatives_skipped": sum(blank.values()),
        "bad_unicode_alternatives_skipped": sum(bad_unicode.values()),
        "fragmented_markup_alternatives_skipped": sum(fragmented_markup.values()),
        "sentence_strategy_counts": [
            {
                "sent_id": sent_id,
                "strategy": strategy,
                "examined": examined[(sent_id, strategy)],
                "loaded": kept[(sent_id, strategy)],
                "blank_skipped": blank[(sent_id, strategy)],
                "bad_unicode_skipped": bad_unicode[(sent_id, strategy)],
                "fragmented_markup_skipped": fragmented_markup[(sent_id, strategy)],
            }
            for sent_id, strategy in keys
        ],
    }
    return alternatives, qa


def observed_target(sentence: dict) -> dict:
    target_sentence = copy.deepcopy(sentence)
    target_sentence["sentence_index"] = 0
    return {
        "id": "target",
        "kind": "observed_target",
        "text": sentence["text"],
        "sentences": [target_sentence],
        "annotation_source": "manual_csv_native_tokens_and_entities",
    }


def annotate_in_batches(
    annotator: SeededCorPipe,
    context: Sequence[dict],
    branches: Sequence[dict],
    job_id: str,
    batch_size: int,
) -> list[dict]:
    outputs = []
    for start in range(0, len(branches), batch_size):
        stop = min(start + batch_size, len(branches))
        outputs.extend(annotator.annotate_branches(context, branches[start:stop], f"{job_id}-{start}-{stop}"))
    return outputs


def annotate_story(
    story_id: int,
    story: list[dict],
    alternatives: dict[int, list[dict]],
    annotator: SeededCorPipe,
    splitter: StanzaSentenceSplitter,
    silver: CachedSilverParser,
    model_name: str,
    segment: int,
    batch_size: int,
    alternative_qa: dict,
) -> dict:
    items = []
    for sent_id in sorted(alternatives):
        if not 0 <= sent_id < len(story):
            raise ValueError(f"story {story_id}: sentence {sent_id} is outside the story")
        item_id = f"naturalstories_{story_id}::s{sent_id}"
        branches = []
        for alternative in alternatives[sent_id]:
            branch = dict(alternative)
            branch["grammatical_sentences"] = parse_text(branch["text"], splitter, silver)
            branches.append(branch)
        annotated = annotate_in_batches(
            annotator, story[:sent_id], branches, item_id, batch_size
        )
        clean_alternatives = []
        for branch in annotated:
            branch = dict(branch)
            branch.pop("context_cluster_ids", None)
            branch.pop("n_given_mentions", None)
            branch.pop("n_new_mentions", None)
            branch["sentences"] = [clean_corpipe_sentence(sentence) for sentence in branch["sentences"]]
            branch["annotation_source"] = "stanza_parse_seeded_corpipe_entities"
            clean_alternatives.append(branch)
        items.append({
            "id": item_id,
            "sentence_index": sent_id,
            "prefix_sentence_count": sent_id,
            "target": observed_target(story[sent_id]),
            "alternatives": clean_alternatives,
        })
        print(f"story {story_id}: sentence {sent_id}, alternatives={len(annotated)}", flush=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "complete": True,
        "dataset": "naturalstories",
        "story_id": f"naturalstories_{story_id}",
        "model": model_name,
        "corpipe_segment": segment,
        "alternative_input_qa": alternative_qa,
        "context": story,
        "items": items,
        "semantics": {
            "manual_story": "CSV tokens, dependencies, mention spans, and cluster IDs kept natively",
            "syntax": "manual CoNLL-X-style CSV for targets/context; Stanza English UD for alternatives only",
            "alternative_entities": "seeded CorPipe; recurrent clusters inherit frozen manual-prefix IDs",
            "branch_isolation": True,
            "right_context_disabled": True,
        },
    }


def run(args: argparse.Namespace) -> None:
    csv_files = sorted(Path(args.entity_dir).glob("story_*_entity_annotated.csv"), key=story_number)
    if args.stories:
        requested = set(args.stories)
        csv_files = [path for path in csv_files if story_number(path) in requested]
    if not csv_files:
        raise FileNotFoundError(f"no annotated story CSVs in {args.entity_dir}")

    # CorPipe/minnt must initialize before Stanza initializes multiprocessing.
    annotator = SeededCorPipe(
        args.corpipe_source,
        args.model,
        segment=args.segment,
        device=args.device,
        threads=args.threads,
    )
    silver = CachedSilverParser(StanzaSilverParser(args.stanza_model_dir, args.stanza_gpu))
    splitter = StanzaSentenceSplitter(args.stanza_model_dir)

    output_dir = Path(args.out_dir) / "naturalstories"
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        story_id = story_number(csv_path)
        output_path = output_dir / f"naturalstories_{story_id}.json"
        if output_path.exists() and not args.overwrite:
            print(f"skip {output_path}", flush=True)
            continue
        alternative_path = Path(args.alternatives_dir) / f"story_{story_id}_filtered.jsonl"
        if not alternative_path.is_file():
            raise FileNotFoundError(alternative_path)
        story = read_manual_story(csv_path, story_id)
        alternatives, qa = load_alternatives(alternative_path, args.branch_limit)
        output = annotate_story(
            story_id, story, alternatives, annotator, splitter, silver,
            args.model, args.segment, args.branch_batch_size, qa,
        )
        atomic_json(output_path, output)
        print(f"wrote {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-dir", required=True)
    parser.add_argument("--alternatives-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--corpipe-source", required=True)
    parser.add_argument("--ns-conllx", help="accepted for old commands; the manual CSV remains authoritative")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--segment", type=int, default=2560)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stanza-model-dir")
    parser.add_argument("--stanza-gpu", action="store_true")
    parser.add_argument("--stories", type=int, nargs="*")
    parser.add_argument("--branch-limit", type=int, default=100)
    parser.add_argument("--branch-batch-size", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.branch_limit <= 0 or args.branch_batch_size <= 0:
        parser.error("branch limits and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
