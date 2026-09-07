#!/usr/bin/env python3
"""Keep the first spaCy sentence from every raw CLASP continuation."""

import argparse
import json
import os
import re
from pathlib import Path


RAW_NAME = re.compile(r"clasp_(\d+)_raw\.jsonl$")


def clasp_id(path):
    """Read the numeric CLASP ID from a raw filename."""
    match = RAW_NAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"unexpected raw filename: {path.name}")
    return int(match.group(1))


def first_sentences(nlp, raws):
    """Strip each raw and retain its first spaCy sentence, preserving blanks."""
    filtered = [raw.strip() for raw in raws]
    positions = [index for index, text in enumerate(filtered) if text]
    texts = [filtered[index] for index in positions]

    for index, doc in zip(positions, nlp.pipe(texts, batch_size=64)):
        sentence = next(doc.sents, None)
        filtered[index] = sentence.text if sentence is not None else doc.text
    return filtered


def write_jsonl_atomic(path, rows):
    """Write the complete filtered ID file before replacing any old copy."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    """Filter every clasp_<ID>_raw.jsonl file in numeric ID order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()

    import spacy

    nlp = spacy.load(args.spacy_model, disable=["ner", "lemmatizer"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(args.raw_dir.glob("clasp_*_raw.jsonl"), key=clasp_id)
    if not raw_files:
        parser.error(f"no clasp_<ID>_raw.jsonl files found in {args.raw_dir}")

    for raw_file in raw_files:
        rows = []
        with raw_file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                raws = row.get("raws", [])
                if len(raws) < args.n:
                    raise ValueError(
                        f"{raw_file}:{line_number} has {len(raws)} raws; "
                        f"cannot keep {args.n}"
                    )
                row["raws"] = first_sentences(nlp, raws[:args.n])
                row["n_raws"] = len(row["raws"])
                rows.append(row)

        output_file = args.output_dir / f"clasp_{clasp_id(raw_file)}_filtered.jsonl"
        write_jsonl_atomic(output_file, rows)
        print(f"wrote {output_file}")


if __name__ == "__main__":
    main()