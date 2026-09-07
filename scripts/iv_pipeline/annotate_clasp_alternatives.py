#!/usr/bin/env python3
"""Silver-parse and coreference-annotate CLASP targets and alternatives.

CLASP has no manual entity layer.  Stanza parses the context, targets, and
alternatives.  CorPipe annotates the context once, then decodes every target
and alternative independently against that frozen context.
"""

from __future__ import annotations

import argparse
import ast
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
FILTERED_NAME = re.compile(r"clasp_(\d+)_filtered\.jsonl$")


def normalise_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def has_bad_unicode(text: str) -> bool:
    return "\ufffd" in text or any(0x80 <= ord(char) <= 0x9F for char in text)


def load_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def filtered_files(path: str | Path) -> list[Path]:
    """Return one filtered file or every filtered file in a directory."""

    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    files = []
    for file_path in path.glob("clasp_*_filtered.jsonl"):
        match = FILTERED_NAME.fullmatch(file_path.name)
        if match:
            files.append((int(match.group(1)), file_path))
    if not files:
        raise FileNotFoundError(f"no clasp_<ID>_filtered.jsonl files in {path}")
    return [file_path for _context_id, file_path in sorted(files)]


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def parse_ratings(value: str) -> list[int]:
    parsed = ast.literal_eval(f"[{value}]") if value.strip() else []
    if not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"non-integer ratings: {value!r}")
    return parsed


def load_ratings(path: str | Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            context_id = int(row["ID"])
            groups[context_id].append({
                "language": row["Language"],
                "sentence": normalise_text(row["Sentence"]),
                "pre_context": normalise_text(row["Pre-Context"]),
                "post_context": normalise_text(row["Post-Context"]),
                "ratings_without_context": parse_ratings(row["Without-Context Ratings"]),
                "ratings_with_context": parse_ratings(row["With-Context Ratings"]),
            })
    expected = ["English", "Czech", "German", "Spanish", "French"]
    for context_id, rows in groups.items():
        if [row["language"] for row in rows] != expected:
            raise ValueError(f"CLASP {context_id}: unexpected language order")
        if len({row["pre_context"] for row in rows}) != 1:
            raise ValueError(f"CLASP {context_id}: targets do not share one context")
    return dict(groups)


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
    return [silver.parse(piece) for piece in splitter.split(text)]


def raw_mention(sentence: dict, mention: dict) -> dict:
    start, end = int(mention["token_start"]), int(mention["token_end"])
    char_start = int(sentence["tokens"][start]["char_start"])
    char_end = int(sentence["tokens"][end]["char_end"])
    return {
        "token_start": start,
        "token_end": end,
        "char_start": char_start,
        "char_end": char_end,
        "text": sentence["text"][char_start:char_end],
        "cluster_id": int(mention["cluster_id"]),
    }


def add_token_entity_columns(sentence: dict) -> dict:
    output = copy.deepcopy(sentence)
    events: list[list[tuple[int, int, str]]] = [[] for _ in output["tokens"]]
    for mention in output.get("mentions", []):
        start, end = int(mention["token_start"]), int(mention["token_end"])
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


def clean_sentence(sentence: dict) -> dict:
    clean = {
        "sentence_index": int(sentence.get("sentence_index", 0)),
        "text": sentence["text"],
        "grammar_source": sentence.get("grammar_source"),
        "tokens": [dict(token) for token in sentence["tokens"]],
    }
    clean["mentions"] = [raw_mention(clean, mention) for mention in sentence.get("mentions", [])]
    validate_sentence_spec(clean)
    return add_token_entity_columns(clean)


def read_generation_groups(path: str | Path, limit: int) -> dict[int, dict]:
    """Merge one record per strategy into one CLASP context record."""

    groups: dict[int, dict] = {}
    for input_file in filtered_files(path):
        for row in load_jsonl(input_file):
            context_id = int(row["clasp_id"])
            context = normalise_text(row["context"])
            targets = [
                {"language": str(target["language"]), "text": normalise_text(target["text"])}
                for target in row["targets"]
            ]
            group = groups.setdefault(context_id, {
                "context": context,
                "targets": targets,
                "alternatives": [],
                "qa": Counter(),
                "strategy_counts": {},
            })
            if group["context"] != context or group["targets"] != targets:
                raise ValueError(f"CLASP {context_id}: context/targets differ across strategies")
            strategy = str(row["strategy"])
            if strategy in group["strategy_counts"]:
                raise ValueError(f"CLASP {context_id}: duplicate strategy {strategy}")
            kept = 0
            for value in row["raws"]:
                if kept >= limit:
                    break
                group["qa"]["examined"] += 1
                text = value.get("text") if isinstance(value, dict) else value
                raw = value.get("raw") if isinstance(value, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"CLASP {context_id}/{strategy}: non-string generation")
                if not text.strip():
                    group["qa"]["blank_skipped"] += 1
                    continue
                if has_bad_unicode(text):
                    group["qa"]["bad_unicode_skipped"] += 1
                    continue
                kept += 1
                group["alternatives"].append({
                    "id": f"{strategy}_{kept}",
                    "kind": "generated_alternative",
                    "strategy": strategy,
                    "sample_index": kept,
                    "text": normalise_text(text),
                    "raw": raw,
                })
            group["strategy_counts"][strategy] = kept
            group["qa"]["loaded"] += kept
    return groups


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
        print(f"  {job_id}: branches {stop}/{len(branches)}", flush=True)
    return outputs


def clean_branch(branch: dict) -> dict:
    output = dict(branch)
    output.pop("context_cluster_ids", None)
    output.pop("n_given_mentions", None)
    output.pop("n_new_mentions", None)
    output["sentences"] = [clean_sentence(sentence) for sentence in branch["sentences"]]
    output["annotation_source"] = "stanza_parse_seeded_corpipe_entities"
    return output


def annotate_group(
    context_id: int,
    group: dict,
    rating_rows: Sequence[dict],
    annotator: SeededCorPipe,
    splitter: StanzaSentenceSplitter,
    silver: CachedSilverParser,
    model_name: str,
    segment: int,
    batch_size: int,
    limit: int,
) -> dict:
    if rating_rows:
        if normalise_text(rating_rows[0]["pre_context"]) != group["context"]:
            raise ValueError(f"CLASP {context_id}: generation context differs from ratings CSV")
        expected_targets = [(row["language"], row["sentence"]) for row in rating_rows]
        actual_targets = [(row["language"], row["text"]) for row in group["targets"]]
        if expected_targets != actual_targets:
            raise ValueError(f"CLASP {context_id}: generated targets differ from ratings CSV")

    parsed_context = parse_text(group["context"], splitter, silver) if group["context"] else []
    context = annotator.annotate_document(parsed_context, f"clasp_{context_id}-context")
    clean_context = [clean_sentence(sentence) for sentence in context]

    target_inputs = []
    for index, target in enumerate(group["targets"]):
        branch = {
            "id": f"target_{target['language']}",
            "kind": "observed_target" if index == 0 else "back_translation_target",
            "language": target["language"],
            "text": target["text"],
            "grammatical_sentences": parse_text(target["text"], splitter, silver),
        }
        target_inputs.append(branch)

    alternative_inputs = []
    for alternative in group["alternatives"]:
        branch = dict(alternative)
        branch["grammatical_sentences"] = parse_text(branch["text"], splitter, silver)
        alternative_inputs.append(branch)

    targets = [
        clean_branch(branch)
        for branch in annotate_in_batches(
            annotator, clean_context, target_inputs, f"clasp_{context_id}-targets", batch_size
        )
    ]
    alternatives = [
        clean_branch(branch)
        for branch in annotate_in_batches(
            annotator, clean_context, alternative_inputs, f"clasp_{context_id}-alternatives", batch_size
        )
    ]

    rating_by_language = {row["language"]: row for row in rating_rows}
    items = []
    for target in targets:
        rating = rating_by_language.get(target["language"], {})
        target.update({
            "ratings_without_context": rating.get("ratings_without_context", []),
            "ratings_with_context": rating.get("ratings_with_context", []),
            "post_context": rating.get("post_context", ""),
        })
        items.append({
            "id": f"clasp_{context_id}::{target['language']}",
            "language": target["language"],
            "sentence_index": len(clean_context),
            "prefix_sentence_count": len(clean_context),
            "target": target,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "complete": True,
        "dataset": "clasp",
        "context_id": f"clasp_{context_id}",
        "model": model_name,
        "corpipe_segment": segment,
        "requested_branch_limit_per_strategy": limit,
        "input_qa": {
            **dict(group["qa"]),
            "strategy_counts": group["strategy_counts"],
        },
        "context_text": group["context"],
        "context": clean_context,
        "items": items,
        "alternatives": alternatives,
        "semantics": {
            "syntax": "Stanza English UD silver for context, targets, and alternatives",
            "context_entities": "one ordinary CorPipe document pass",
            "branches": "each target and alternative decoded independently against the same frozen context",
            "branch_isolation": True,
            "right_context_disabled": True,
        },
    }


def run(args: argparse.Namespace) -> None:
    groups = read_generation_groups(args.clasp_alternatives, args.branch_limit)
    ratings = load_ratings(args.clasp_gold)
    requested = set(args.clasp_ids or groups)
    missing = requested - set(groups)
    if missing:
        raise ValueError(f"CLASP IDs absent from alternatives: {sorted(missing)}")

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

    output_dir = Path(args.out_dir) / "clasp"
    output_dir.mkdir(parents=True, exist_ok=True)
    for context_id in sorted(requested):
        output_path = output_dir / f"clasp_{context_id}.json"
        if output_path.exists() and not args.overwrite:
            print(f"skip {output_path}", flush=True)
            continue
        output = annotate_group(
            context_id,
            groups[context_id],
            ratings.get(context_id, []),
            annotator,
            splitter,
            silver,
            args.model,
            args.segment,
            args.branch_batch_size,
            args.branch_limit,
        )
        atomic_json(output_path, output)
        print(f"wrote {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clasp-alternatives",
        required=True,
        help="one filtered JSONL or a directory of clasp_<ID>_filtered.jsonl files",
    )
    parser.add_argument("--clasp-gold", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--corpipe-source", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--segment", type=int, default=2560)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stanza-model-dir")
    parser.add_argument("--stanza-gpu", action="store_true")
    parser.add_argument("--clasp-ids", type=int, nargs="*")
    parser.add_argument("--branch-limit", type=int, default=100)
    parser.add_argument("--branch-batch-size", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.branch_limit <= 0 or args.branch_batch_size <= 0:
        parser.error("branch limits and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
