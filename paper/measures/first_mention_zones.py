#!/usr/bin/env python3
"""Natural Stories only: the reading-time zones (whitespace words) of the target's first mention,
for the word-level localisation of Appendix A.

The annotation tokens (Stanza) and the reading-time zones do not align one to one, so each token
is mapped to a zone by cumulative character offsets after stripping everything but letters and
digits. The zone map (item, sentence_id, zone, word) is taken from the released word-level
surprisal file of the repository (annotated/surprisal/pythia-70m-deduped-step1000-full.tsv),
whose sentence_id is 1-based while the annotation's sentence_index is 0-based.

One row per target: the zone of the head of its first mention, the zone of the last token of that
mention, the mention's form and whether it refers to an established or a new entity.

Usage: first_mention_zones.py <zone map tsv> <annotated json or directory> <out_csv>
"""

import csv
import re
import sys

from common import iter_records, mentions_of, record_key, target_sentences, write_csv

NOT_ALNUM = re.compile(r"[^A-Za-z0-9]")


def normalise(text):
    return NOT_ALNUM.sub("", text).lower()


def load_zone_map(path):
    zones = {}
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (int(row["item"]), int(row["sentence_id"]))
            zones.setdefault(key, []).append((int(row["zone"]), row["word"].strip()))
    return {key: sorted(values) for key, values in zones.items()}


def token_zones(zone_words, tokens):
    """Zone of every token: the zone whose character span contains the token's midpoint."""
    spans = []
    offset = 0
    for zone, word in zone_words:
        length = len(normalise(word))
        spans.append((offset, offset + length, zone))
        offset += length
    out = []
    offset = 0
    for token in tokens:
        length = len(normalise(token["text"]))
        midpoint = offset + max(0, length - 1) / 2
        out.append(next((zone for start, end, zone in spans if start <= midpoint < end), None))
        offset += length
    return out


def main(zone_map_tsv, inputs, out_csv):
    zone_map = load_zone_map(zone_map_tsv)
    rows = []
    for record in iter_records([inputs]):
        story = int(record["document_id"].split("_")[1])
        sentence_id = record["sentence_index"] + 1
        sentences = target_sentences(record)
        mentions = mentions_of(sentences)
        row = record_key(record)
        if mentions:
            first = mentions[0]
            offset = sum(len(sentence["tokens"]) for sentence in sentences[:first.sentence])
            tokens = [token for sentence in sentences for token in sentence["tokens"]]
            zones = token_zones(zone_map.get((story, sentence_id), []), tokens)
            established = {m.entity for m in mentions_of(record["context_sentences"])}
            row.update(
                first_zone_head=zones[offset + first.head],
                first_zone_end=zones[offset + first.token_end],
                first_form=first.form,
                first_kind="established" if first.entity in established else "new",
            )
        rows.append(row)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
