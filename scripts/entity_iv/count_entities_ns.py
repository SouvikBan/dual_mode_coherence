#!/usr/bin/env python3
"""Per-sentence AND cumulative entity counts for Natural Stories.

For a sentence k, the context is sentences 0..k-1 and the sentence itself is k.
Two kinds of count are produced for every sentence:

  per-sentence   what is in sentence k alone
  cumulative     summed over every entity established up to and including k,
                 i.e. over sentences 0..k

"The sum of the number of mentions of each entity established till that point"
is, summed over entities, simply the running total of mentions, so
cum_n_mentions is that quantity. cum_n_transitions is the same running total
restricted to mentions that are not the first mention of their entity.

Both an inclusive (0..k) and an exclusive (0..k-1, the prior context only)
version are written, because which one is wanted depends on whether the
sentence being read is allowed to count towards its own predictor.

A sentence with no entity mentions at all is kept, with its per-sentence counts
at 0 and the cumulative totals carried forward. Three such sentences exist in
Natural Stories (story 3 s23, story 4 s38 and s42); they are genuine, being
interjections or a pleonastic "it".

    python count_entities_ns.py \
        --mention-costs mention_costs/naturalstories/mention_costs.csv \
        --entity-dir annotated_csv \
        --out naturalstories_entity_counts.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FIELDS = ("item sentence_id sentence_index_0based "
          "n_mentions n_transitions n_first_mentions n_entities "
          "n_entities_from_context "
          "role_cost_sum infstat_cost_sum "
          "cum_n_mentions cum_n_transitions cum_n_entities "
          "cum_mentions_per_entity "
          "cum_role_cost cum_infstat_cost "
          "cum_role_cost_per_transition cum_infstat_cost_per_transition "
          "ctx_n_mentions ctx_n_transitions ctx_n_entities "
          "ctx_role_cost ctx_infstat_cost "
          "ctx_role_cost_per_transition ctx_infstat_cost_per_transition").split()


def _cost(value):
    """A blank cost means the mention has no previous mention, so there is no
    transition to score. It is skipped, never counted as 0 bits."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sentence_counts(path):
    """story -> sentence_index -> list of (entity_id, is_first, role, infstat)."""
    per = defaultdict(lambda: defaultdict(list))
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            story = int("".join(c for c in row["document_id"] if c.isdigit()))
            per[story][int(row["sentence_index"])].append(
                (int(row["entity_id"]), int(row["is_first_mention"]),
                 _cost(row["role_transition_cost"]),
                 _cost(row["infstat_transition_cost"])))
    return per


def story_lengths(entity_dir):
    """story -> number of sentences, so sentences with no mentions are kept."""
    lengths = {}
    for path in Path(entity_dir).glob("story_*_entity_annotated.csv"):
        story = int("".join(c for c in path.stem.split("_")[1] if c.isdigit()))
        with path.open(encoding="utf-8-sig") as stream:
            ids = {int(r["sent_id"]) for r in csv.DictReader(stream, delimiter=";")}
        lengths[story] = max(ids) + 1 if min(ids) == 0 else max(ids)
    return lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mention-costs", required=True)
    parser.add_argument("--entity-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per = sentence_counts(args.mention_costs)
    lengths = story_lengths(args.entity_dir)
    rows = []
    for story in sorted(lengths):
        cum_m = cum_t = 0
        cum_role = cum_inf = 0.0
        seen = set()
        for k in range(lengths[story]):
            mentions = per[story].get(k, [])
            # totals for the prior context alone, before this sentence is added
            ctx = dict(ctx_n_mentions=cum_m, ctx_n_transitions=cum_t,
                       ctx_n_entities=len(seen),
                       ctx_role_cost=round(cum_role, 6),
                       ctx_infstat_cost=round(cum_inf, 6),
                       ctx_role_cost_per_transition=(round(cum_role / cum_t, 6)
                                                     if cum_t else 0.0),
                       ctx_infstat_cost_per_transition=(round(cum_inf / cum_t, 6)
                                                        if cum_t else 0.0))
            # the entities the context had established, BEFORE this sentence
            # is folded into `seen`
            established = set(seen)
            n_first = sum(f for _, f, _, _ in mentions)
            n_trans = len(mentions) - n_first
            # a sentence's own cost is the sum over its mentions that HAVE a
            # transition, matching how summed token surprisal is formed
            role = sum(r for _, _, r, _ in mentions if r is not None)
            inf = sum(i for _, _, _, i in mentions if i is not None)
            cum_m += len(mentions)
            cum_t += n_trans
            cum_role += role
            cum_inf += inf
            seen.update(e for e, _, _, _ in mentions)
            rows.append({"item": story, "sentence_id": k + 1,
                         "sentence_index_0based": k,
                         "n_mentions": len(mentions), "n_transitions": n_trans,
                         "n_first_mentions": n_first,
                         "n_entities": len({e for e, _, _, _ in mentions}),
                         # entities in this sentence that the context had
                         # already established, the CLASP file's counterpart
                         "n_entities_from_context":
                             len({e for e, _, _, _ in mentions} & established),
                         "role_cost_sum": round(role, 6),
                         "infstat_cost_sum": round(inf, 6),
                         "cum_n_mentions": cum_m, "cum_n_transitions": cum_t,
                         "cum_n_entities": len(seen),
                         "cum_mentions_per_entity": (cum_m / len(seen)) if seen else 0.0,
                         "cum_role_cost": round(cum_role, 6),
                         "cum_infstat_cost": round(cum_inf, 6),
                         "cum_role_cost_per_transition": round(cum_role / cum_t, 6) if cum_t else 0.0,
                         "cum_infstat_cost_per_transition": round(cum_inf / cum_t, 6) if cum_t else 0.0,
                         **ctx})
    with Path(args.out).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} sentences, {len(lengths)} stories")
    empty = [r for r in rows if r["n_mentions"] == 0]
    print(f"sentences with no mentions (kept, cumulative carried forward): "
          f"{[(r['item'], r['sentence_id']) for r in empty]}")


if __name__ == "__main__":
    main()
