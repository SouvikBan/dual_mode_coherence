#!/usr/bin/env python3
"""Generate one shared alternative pool for the five targets of each CLASP ID."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from vllm_common import (
    add_generation_args, create_llm, encode_context, run_workers,
    sample_raws, validate_args, write_jsonl_atomic,
)


LANGUAGES = ("English", "Czech", "German", "Spanish", "French")
REQUIRED_COLUMNS = {"ID", "Language", "Sentence", "Pre-Context"}


def clean(text):
    return " ".join((text or "").split())


def load_clasp(path, allow_partial=False):
    """Use the same ID, language, target, and context validation as the original."""
    groups = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                clasp_id = int(row["ID"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"line {line_number}: invalid ID {row.get('ID')!r}") from error
            language = clean(row["Language"])
            sentence = clean(row["Sentence"])
            context = clean(row["Pre-Context"])
            if not language or not sentence:
                raise ValueError(f"line {line_number}: empty Language or Sentence")
            groups[clasp_id].append((language, sentence, context))

    if not allow_partial and set(groups) != set(range(100)):
        missing = sorted(set(range(100)) - set(groups))
        extra = sorted(set(groups) - set(range(100)))
        raise ValueError(f"expected IDs 0..99; missing={missing}, extra={extra}")

    records = []
    for clasp_id in sorted(groups):
        rows = groups[clasp_id]
        by_language = {language: (sentence, context) for language, sentence, context in rows}
        if len(rows) != 5 or set(by_language) != set(LANGUAGES):
            found = [language for language, _, _ in rows]
            raise ValueError(f"ID {clasp_id}: expected exactly {list(LANGUAGES)}, found {found}")
        contexts = {context for _, context in by_language.values()}
        if len(contexts) != 1:
            raise ValueError(f"ID {clasp_id}: the five rows have different contexts")
        records.append({
            "clasp_id": clasp_id,
            "context": contexts.pop(),
            "targets": [
                {"language": language, "text": by_language[language][0]}
                for language in LANGUAGES
            ],
        })
    return records


def generate_rows(llm, record, args, gpu):
    token_ids = encode_context(llm, record["context"], args.max_new_tokens)
    for strategy in args.strategies:
        raws = sample_raws(llm, token_ids, strategy,
                           f"clasp:{record['clasp_id']}", args, strip=True)
        yield {
            "clasp_id": record["clasp_id"],
            "context": record["context"],
            "targets": record["targets"],
            "strategy": strategy,
            "raws": raws,
        }
        print(f"[GPU {gpu}] CLASP {record['clasp_id']} {strategy}: {len(raws)} raws", flush=True)


def worker(gpu, records, args):
    llm = create_llm(gpu, args)
    for record in records:
        path = args.output_dir / f"clasp_{record['clasp_id']}_raw.jsonl"
        write_jsonl_atomic(path, generate_rows(llm, record, args, gpu))
        print(f"[GPU {gpu}] wrote {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clasp-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    add_generation_args(parser)
    args = parser.parse_args()
    validate_args(parser, args)

    records = load_clasp(args.clasp_data, args.allow_partial)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        records = [r for r in records if not (
            args.output_dir / f"clasp_{r['clasp_id']}_raw.jsonl"
        ).exists()]
    print(f"{len(records)} IDs x {len(args.strategies)} strategies x {args.samples} raws", flush=True)
    run_workers(worker, records, args)


if __name__ == "__main__":
    main()
