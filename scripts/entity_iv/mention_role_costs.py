#!/usr/bin/env python3
"""Per-mention transition costs for the mentions actually in an utterance.

How this differs from the earlier gold-count measure
----------------------------------------------------
The superseded count_gold_entities.py in the research checkout snapshots, at every sentence, EVERY entity seen so far
and re-reports whatever its latest transition was. Measured on Natural Stories
that means 93% of the scored rows are entities absent from the sentence, the
transition being re-reported is on average 15.4 sentences old, and 73% of the
values are structural zeros from entities that have had no transition at all.
The result correlates r = 0.92 with position in the story.

This script scores only the mentions that are IN the utterance. For each one:

    role transition cost    = -log2 P(role of THIS mention | role of the same
                              entity's PREVIOUS mention), from the GUM model
    infstat transition cost = the same for information status

There is no summation over mentions. A sentence is described by the per-mention
costs themselves (long table) and, in the wide table, by their mean and max
together with how many mentions the sentence has. A sum would grow with the
mention count and re-measure sentence length; keeping the count as its own
variable lets a model use it explicitly.

A mention that is its entity's FIRST mention has no transition. It is counted
in n_mentions but excluded from the cost aggregates and flagged
is_first_mention, never scored as 0 bits.

    python mention_role_costs.py \\
        --ns-csv-dir annotated_csv \\
        --clasp-json-dir annotated_alternatives/clasp \\
        --role-costs gum_deprel_costs_50.json --role-inventory deprel \\
        --infstat-costs gum_infstat_costs.json \\
        --out-dir mention_costs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from entity_roles import INVENTORIES, mention_head, mention_role, project_state  # noqa: E402
from read_ns_annotation import INFSTATS, load_ns_csvs  # noqa: E402

LONG_FIELDS = ("dataset document_id item_id language section sentence_index mention_index "
               "entity_id mention_text is_first_mention sentences_since_last "
               "previous_role role role_transition_cost "
               "previous_infstat infstat infstat_transition_cost").split()


def clasp_from_annotation(directory):
    """CLASP context and targets from the annotated JSONs written by stage 2.

    These carry full Stanza syntax on every token, so role transitions are
    computable. They carry no information status on context mentions and only a
    binary given/new on targets, which does not map onto the six GUM states, so
    the information-status channel is unavailable from this source; use
    --clasp-annotation-dir (the manual token CSVs) for that.
    """
    for path in sorted(Path(directory).glob("clasp_*.json"),
                       key=lambda p: int(p.stem.split("_")[-1])):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("dataset") != "clasp":
            continue
        yield document


def load_costs(path, expected=None):
    with Path(path).open(encoding="utf-8") as stream:
        stored = json.load(stream)
    costs = stored["transition_costs"]
    states = tuple(stored.get("roles") or stored.get("states") or costs)
    for previous in states:
        for current in states:
            value = float(costs[previous][current])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: bad cost {previous}->{current}")
    if expected and stored.get("role_inventory") and stored["role_inventory"] != expected:
        raise SystemExit(f"{path} was fitted with role_inventory "
                         f"{stored['role_inventory']!r}, not {expected!r}")
    return costs, states


def score_document(sentences, meta, role_costs, role_states, infstat_costs, inventory):
    """One row per mention, in reading order."""
    last_role, last_infstat, last_sentence = {}, {}, {}
    rows = []
    for index, sentence in enumerate(sentences):
        tokens = sentence["tokens"]
        ordered = sorted(sentence.get("mentions", []),
                         key=lambda m: (int(m["token_start"]), -int(m["token_end"]),
                                        int(m["cluster_id"])))
        for position, mention in enumerate(ordered):
            entity = int(mention["cluster_id"])
            start, end = int(mention["token_start"]), int(mention["token_end"])
            role = project_state(mention_role(tokens, mention_head(tokens, start, end),
                                              inventory), role_states)
            status = mention.get("infstat")
            status = status if status in INFSTATS else None
            first = entity not in last_role
            row = {**meta, "sentence_index": index, "mention_index": position,
                   "entity_id": entity,
                   "mention_text": " ".join(t["text"] for t in tokens[start:end + 1]),
                   "is_first_mention": int(first),
                   "sentences_since_last": "" if first else index - last_sentence[entity],
                   "previous_role": "" if first else last_role[entity],
                   "role": role,
                   "role_transition_cost": ("" if first
                                            else role_costs[last_role[entity]][role]),
                   "previous_infstat": "" if first or last_infstat.get(entity) is None
                                       else last_infstat[entity],
                   "infstat": status or "",
                   "infstat_transition_cost": ""}
            if (not first and status is not None
                    and last_infstat.get(entity) is not None and infstat_costs):
                row["infstat_transition_cost"] = infstat_costs[last_infstat[entity]][status]
            rows.append(row)
            last_role[entity] = role
            last_sentence[entity] = index
            if status is not None:
                last_infstat[entity] = status
    return rows


def summarise(rows):
    """One row per utterance: counts, and mean/max of the costs. No sums."""
    by_sentence = {}
    for row in rows:
        key = (row["dataset"], row["document_id"], row["item_id"], row["language"],
               row["section"], row["sentence_index"])
        by_sentence.setdefault(key, []).append(row)
    out = []
    for key, group in by_sentence.items():
        record = dict(zip(("dataset", "document_id", "item_id", "language",
                           "section", "sentence_index"), key))
        record["n_mentions"] = len(group)
        record["n_entities"] = len({r["entity_id"] for r in group})
        record["n_first_mentions"] = sum(r["is_first_mention"] for r in group)
        for channel in ("role", "infstat"):
            values = [float(r[f"{channel}_transition_cost"]) for r in group
                      if r[f"{channel}_transition_cost"] != ""]
            record[f"n_{channel}_transitions"] = len(values)
            record[f"{channel}_transition_cost_mean"] = (sum(values) / len(values)
                                                         if values else "")
            record[f"{channel}_transition_cost_max"] = max(values) if values else ""
        out.append(record)
    return out


def write(path, rows, fields=None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ns-csv-dir")
    parser.add_argument("--clasp-json-dir",
                        help="annotated_alternatives/clasp written by annotate_clasp.py; full Stanza "
                             "syntax so roles work, but no 6-way information status")
    parser.add_argument("--role-costs", required=True)
    parser.add_argument("--role-inventory", choices=tuple(INVENTORIES), default="deprel")
    parser.add_argument("--infstat-costs")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--stanza-model-dir")
    parser.add_argument("--stanza-gpu", action="store_true")
    args = parser.parse_args()
    if not (args.ns_csv_dir or args.clasp_json_dir):
        parser.error("give --ns-csv-dir and/or --clasp-json-dir")

    role_costs, role_states = load_costs(args.role_costs, args.role_inventory)
    infstat_costs = load_costs(args.infstat_costs)[0] if args.infstat_costs else None
    print(f"role states: {len(role_states)} ({args.role_inventory}); "
          f"information status: {'on' if infstat_costs else 'off'}")

    if args.ns_csv_dir:
        rows = []
        for document in load_ns_csvs(Path(args.ns_csv_dir)):
            story = str(document["story_id"])
            meta = {"dataset": "naturalstories", "document_id": story,
                    "item_id": story, "language": "English", "section": "document"}
            rows += score_document(document["context"], meta, role_costs, role_states,
                                   infstat_costs, args.role_inventory)
        write(args.out_dir / "naturalstories" / "mention_costs.csv", rows, LONG_FIELDS)
        write(args.out_dir / "naturalstories" / "sentence_costs.csv", summarise(rows))

    if args.clasp_json_dir:
        if infstat_costs:
            print("note: the annotated JSONs carry no 6-way information status; "
                  "only the role channel is produced from this source", flush=True)
        rows = []
        for document in clasp_from_annotation(args.clasp_json_dir):
            context_id = str(document["context_id"])
            meta = {"dataset": "clasp", "document_id": context_id,
                    "item_id": f"{context_id}::context", "language": "shared",
                    "section": "context"}
            rows += score_document(document["context"], meta, role_costs, role_states,
                                   None, args.role_inventory)
            for item in document["items"]:
                meta = {"dataset": "clasp", "document_id": context_id,
                        "item_id": item["id"], "language": item["language"],
                        "section": "target"}
                whole = list(document["context"]) + list(item["target"]["sentences"])
                scored = score_document(whole, meta, role_costs, role_states, None,
                                        args.role_inventory)
                rows += [r for r in scored
                         if r["sentence_index"] >= len(document["context"])]
        write(args.out_dir / "clasp" / "mention_costs.csv", rows, LONG_FIELDS)
        write(args.out_dir / "clasp" / "sentence_costs.csv", summarise(rows))


if __name__ == "__main__":
    main()
