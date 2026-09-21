#!/usr/bin/env python3
"""Per-target and cumulative entity counts for CLASP.

CLASP differs structurally from Natural Stories. A context is shared by five
single-sentence targets, one per language, and cluster ids are continuous
across the two, so an entity introduced in the context can recur in a target.
That gives three levels rather than a running total over a document:

    ctx_*   the shared context alone. CONSTANT across the five languages of a
            context, so it varies only between contexts (100 values).
    n_*     the target sentence alone.
    cum_*   context + target, i.e. everything established once the target has
            been read. Varies by language within a context.

`ctx_*` being constant within a context matters for modelling: with a
`(1|context_id)` random intercept those columns are collinear with the
grouping and can only be estimated from between-context variation.

Costs are summed over the mentions that have a transition; a mention with no
previous mention is skipped, never scored 0 bits. The rate forms divide by the
number of transitions, which is what worked on Natural Stories.

    python count_entities_clasp.py \
        --mention-costs mention_costs/clasp/mention_costs.csv \
        --out count_entities_clasp.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FIELDS = ("item_id context_id language "
          "n_mentions n_transitions n_first_mentions n_entities "
          "n_entities_from_context n_role_cost n_infstat_cost "
          "ctx_n_mentions ctx_n_transitions ctx_n_entities "
          "ctx_role_cost ctx_infstat_cost "
          "ctx_role_cost_per_transition ctx_infstat_cost_per_transition "
          "cum_n_mentions cum_n_transitions cum_n_entities "
          "cum_role_cost cum_infstat_cost "
          "cum_role_cost_per_transition cum_infstat_cost_per_transition "
          "cum_mentions_per_entity").split()


def _cost(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mention-costs", required=True)
    parser.add_argument("--annotation-dir",
                        help="annotated_alternatives/clasp; used only to enumerate targets, so "
                             "that a target in which the annotator found NO mentions still gets "
                             "a row of zeros instead of vanishing (clasp_67::Spanish is one). "
                             "Natural Stories keeps its three mention-less sentences the same way.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    context = defaultdict(list)     # document -> mentions of the shared context
    targets = defaultdict(list)     # (document, language, item_id) -> mentions
    with Path(args.mention_costs).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            entry = (int(row["entity_id"]), int(row["is_first_mention"]),
                     _cost(row["role_transition_cost"]),
                     _cost(row["infstat_transition_cost"]))
            if row["section"] == "context":
                context[row["document_id"]].append(entry)
            else:
                targets[(row["document_id"], row["language"], row["item_id"])].append(entry)

    if args.annotation_dir:
        import json
        for path in sorted(Path(args.annotation_dir).glob("clasp_*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("dataset") != "clasp":
                continue
            # context_id is already "clasp_N", not the bare number
            key_doc = str(document["context_id"])
            for item in document.get("items", []):
                key = (key_doc, item["language"], item["id"])
                targets.setdefault(key, [])

    rows = []
    for (document, language, item_id), mentions in sorted(targets.items()):
        ctx = context.get(document, [])
        ctx_entities = {e for e, _, _, _ in ctx}
        ctx_trans = sum(1 for _, f, _, _ in ctx if not f)
        ctx_role = sum(r for _, _, r, _ in ctx if r is not None)
        ctx_inf = sum(i for _, _, _, i in ctx if i is not None)

        n_first = sum(f for _, f, _, _ in mentions)
        n_trans = len(mentions) - n_first
        n_role = sum(r for _, _, r, _ in mentions if r is not None)
        n_inf = sum(i for _, _, _, i in mentions if i is not None)
        t_entities = {e for e, _, _, _ in mentions}

        cum_m = len(ctx) + len(mentions)
        cum_t = ctx_trans + n_trans
        cum_e = len(ctx_entities | t_entities)
        cum_role = ctx_role + n_role
        cum_inf = ctx_inf + n_inf
        rows.append({
            "item_id": item_id, "context_id": int(document.split("_")[-1]),
            "language": language,
            "n_mentions": len(mentions), "n_transitions": n_trans,
            "n_first_mentions": n_first, "n_entities": len(t_entities),
            "n_entities_from_context": len(t_entities & ctx_entities),
            "n_role_cost": round(n_role, 6), "n_infstat_cost": round(n_inf, 6),
            "ctx_n_mentions": len(ctx), "ctx_n_transitions": ctx_trans,
            "ctx_n_entities": len(ctx_entities),
            "ctx_role_cost": round(ctx_role, 6), "ctx_infstat_cost": round(ctx_inf, 6),
            "ctx_role_cost_per_transition": round(ctx_role / ctx_trans, 6) if ctx_trans else 0.0,
            "ctx_infstat_cost_per_transition": round(ctx_inf / ctx_trans, 6) if ctx_trans else 0.0,
            "cum_n_mentions": cum_m, "cum_n_transitions": cum_t, "cum_n_entities": cum_e,
            "cum_role_cost": round(cum_role, 6), "cum_infstat_cost": round(cum_inf, 6),
            "cum_role_cost_per_transition": round(cum_role / cum_t, 6) if cum_t else 0.0,
            "cum_infstat_cost_per_transition": round(cum_inf / cum_t, 6) if cum_t else 0.0,
            "cum_mentions_per_entity": round(cum_m / cum_e, 6) if cum_e else 0.0,
        })

    with Path(args.out).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} targets, "
          f"{len({r['context_id'] for r in rows})} contexts")


if __name__ == "__main__":
    main()
