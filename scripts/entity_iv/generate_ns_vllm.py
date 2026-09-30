#!/usr/bin/env python3
"""Generate Natural Stories alternatives from sentence-ID contexts using vLLM."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from vllm_common import (
    add_generation_args, create_llm, encode_context, run_workers,
    sample_raws, validate_args, write_jsonl_atomic,
)


def story_number(path):
    return int(re.search(r"story_(\d+)_", path.name).group(1))


def read_sentences(path):
    sentences = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            sentences[int(row["sent_id"])].append(row["form"])
    sent_ids = sorted(sentences)
    if sent_ids != list(range(len(sent_ids))):
        raise ValueError(f"{path}: sent_id values must start at 0 and be contiguous")
    return [(sent_id, " ".join(sentences[sent_id])) for sent_id in sent_ids]


def make_contexts(sentences):
    previous = []
    for sent_id, target in sentences:
        yield sent_id, " " if sent_id == 0 else " ".join(previous)
        previous.append(target)


def generate_rows(llm, path, args, gpu):
    story_id = story_number(path)
    for sent_id, context in make_contexts(read_sentences(path)):
        token_ids = encode_context(llm, context, args.max_new_tokens)
        for strategy in args.strategies:
            raws = sample_raws(llm, token_ids, strategy,
                               f"naturalstories:{story_id}:{sent_id}", args)
            yield {"sent_id": sent_id, "strategy": strategy, "raws": raws}
            print(f"[GPU {gpu}] story {story_id} sentence {sent_id} {strategy}: "
                  f"{len(raws)} raws", flush=True)


def worker(gpu, csv_paths, args):
    llm = create_llm(gpu, args)
    for path in csv_paths:
        output_path = args.output_dir / f"story_{story_number(path)}_raw.jsonl"
        write_jsonl_atomic(output_path, generate_rows(llm, path, args, gpu))
        print(f"[GPU {gpu}] wrote {output_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_generation_args(parser)
    args = parser.parse_args()
    validate_args(parser, args)

    paths = sorted(args.csv_dir.glob("story_*_entity_annotated.csv"), key=story_number)
    if not paths:
        raise FileNotFoundError(f"no annotated story CSVs in {args.csv_dir}")
    story_ids = [story_number(path) for path in paths]
    if len(set(story_ids)) != len(story_ids):
        raise ValueError("Multiple input CSVs have the same story ID")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        paths = [p for p in paths if not (
            args.output_dir / f"story_{story_number(p)}_raw.jsonl"
        ).exists()]
    print(f"{len(paths)} stories; {len(args.strategies)} strategies x "
          f"{args.samples} raws per context", flush=True)
    run_workers(worker, paths, args)


if __name__ == "__main__":
    main()
