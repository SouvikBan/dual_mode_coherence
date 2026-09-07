#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import spacy


def story_number(path):
    return int(re.search(r"story_(\d+)_raw\.jsonl$", path.name).group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(
        args.raw_dir.glob("story_*_raw.jsonl"),
        key=story_number,
    )

    for raw_file in raw_files:
        alternatives = {}

        with raw_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue

                row = json.loads(line)
                key = int(row["sent_id"]), row["strategy"]
                alternatives.setdefault(key, [])

                for raw in row["raws"][:args.n]:
                    alternative = raw.strip()
                    if alternative:
                        alternative = next(nlp(alternative).sents).text
                    alternatives[key].append(alternative)

        output_file = args.output_dir / (
            f"story_{story_number(raw_file)}_filtered.jsonl"
        )
        with output_file.open("w", encoding="utf-8") as stream:
            for (sent_id, strategy), raws in sorted(alternatives.items()):
                stream.write(json.dumps({
                    "sent_id": sent_id,
                    "strategy": strategy,
                    "n_raws": len(raws),
                    "raws": raws,
                }, ensure_ascii=False) + "\n")

        print(f"wrote {output_file}")


if __name__ == "__main__":
    main()