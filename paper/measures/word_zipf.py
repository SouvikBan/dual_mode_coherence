#!/usr/bin/env python3
"""Zipf word frequencies (wordfreq, English) for the Natural Stories words, a control of the
word-level model of Appendix A.

Usage: word_zipf.py <all_stories.tok (word, zone, item)> <out_csv>
"""

import csv
import sys

from wordfreq import zipf_frequency


def main(words_tsv, out_csv):
    rows = []
    with open(words_tsv) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append({"item": int(row["item"]), "zone": int(row["zone"]), "word": row["word"],
                         "zipf": zipf_frequency(row["word"], "en")})
    rows.sort(key=lambda row: (row["item"], row["zone"]))
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "zone", "word", "zipf"])
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out_csv, len(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
