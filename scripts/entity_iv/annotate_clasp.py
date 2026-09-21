#!/usr/bin/env python3
"""CLASP: gold context and targets, plus N annotated alternatives per strategy.

Keep beside annotate_common.py and corpipe26_seeded.py.

Context and target entities come from the manual CLASP annotation
(contexts_tokens.csv and processing_tokens.csv). Their syntax is Stanza UD on
the gold tokens, because the deprels in those files are spaCy (ClearNLP)
labels that the IV script rejects. The alternatives are selected and
annotated as described in annotate_common.py; CorPipe links them to the frozen
gold context clusters.

One job = one CLASP ID and one strategy. Finished jobs are saved as part
files (OUT/clasp/parts/clasp_N/), so an interrupted run resumes and several
workers can share the IDs (see run_parallel.py). The ID file is assembled
when all its strategies are done.
"""

import argparse
import ast
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from annotate_common import (
    AnnotationCache, LastUsed, WorkQueue, add_common_args, align_offsets, annotate_pool,
    annotate_target, atomic_json, build_gold_sentence, check_args, check_iv_labels, corpipe_context,
    drop_space_tokens, entity_spans, initialise, load_jsonl, normalise_text, read_json,
    read_token_csv, run_jobs, settings_hash, summary_line,
)

LANGUAGES = ("English", "Czech", "German", "Spanish", "French")


def load_ratings(path):
    import csv
    groups = defaultdict(dict)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key, language = int(row["ID"]), row["Language"]
            if language in groups[key]:
                raise ValueError(f"CLASP {key}: duplicate language {language}")
            groups[key][language] = row
    return groups


def ratings_list(value):
    value = (value or "").strip()
    result = ast.literal_eval(value if value.startswith("[") else f"[{value}]") if value else []
    if not isinstance(result, list) or not all(isinstance(x, int) for x in result):
        raise ValueError("ratings must be a list of integers")
    return result


def load_gold(annotation_dir):
    """Rows of contexts_tokens.csv by id/sentence, processing_tokens.csv by id/language."""
    directory = Path(annotation_dir)
    contexts = defaultdict(lambda: defaultdict(list))
    for row in read_token_csv(directory / "contexts_tokens.csv", escaped=True):
        contexts[int(row["id"])][int(row["sent_id"])].append(row)
    targets = defaultdict(list)
    for row in read_token_csv(directory / "processing_tokens.csv", escaped=True):
        targets[(int(row["id"]), row["language"])].append(row)
    for group in [*targets.values(), *(s for c in contexts.values() for s in c.values())]:
        group.sort(key=lambda row: int(row["token_id"]))
    return contexts, targets


def gold_context(clasp_id, context_text, sentences, parser):
    """Gold context sentences; texts are cut from the generation context."""
    sent_ids = sorted(sentences)
    if sent_ids != list(range(len(sent_ids))):
        raise ValueError(f"clasp_{clasp_id}: context sent_id values must be 0..n-1")
    rows, spans = [], []
    for s in sent_ids:
        kept, kept_spans = drop_space_tokens(
            sentences[s], entity_spans([row["entity_layer"] for row in sentences[s]]))
        rows.append(kept)
        spans.append(kept_spans)
    forms = [[row["form"] for row in sentence] for sentence in rows]
    offsets = align_offsets(context_text, [form for sentence in forms for form in sentence])
    parsed = parser.parse_pretokenized(forms)
    output, cursor = [], 0
    for index, sentence_rows in enumerate(rows):
        text = relative = None
        if offsets:
            span = offsets[cursor:cursor + len(sentence_rows)]
            base = span[0][0]
            text = context_text[base:span[-1][1]]
            relative = [(start - base, end - base) for start, end in span]
        cursor += len(sentence_rows)
        sentence = build_gold_sentence(
            forms[index], text, relative, spans[index], words_by_token=parsed[index], token_ids=[row["token_id"] for row in sentence_rows],
            meta={"sentence_index": index, "source_document_id": f"clasp_{clasp_id}",
                  "source_sentence_index": index})
        check_iv_labels(sentence, f"clasp_{clasp_id} context sentence {index}")
        output.append(sentence)
    return output, offsets is not None


def gold_target(clasp_id, language, text, rows, parser):
    rows, spans = drop_space_tokens(rows, entity_spans([row["entity_layer"] for row in rows]))
    forms = [row["form"] for row in rows]
    offsets = align_offsets(text, forms)
    parsed = parser.parse_pretokenized([forms])[0]
    sentence = build_gold_sentence(
        forms, text if offsets else None, offsets, spans,
        words_by_token=parsed, token_ids=[row["token_id"] for row in rows],
        meta={"sentence_index": 0, "source_document_id": f"clasp_{clasp_id}", "language": language})
    check_iv_labels(sentence, f"clasp_{clasp_id} target {language}")
    return sentence, offsets is not None


def check_rows(rows, ratings, args):
    reference = rows[0]
    job_id = f"clasp_{int(reference['clasp_id'])}"
    context_text = normalise_text(reference["context"])
    targets = reference["targets"]
    if [t["language"] for t in targets] != list(LANGUAGES) or set(ratings) != set(LANGUAGES):
        raise ValueError(f"{job_id}: expected five CLASP targets/ratings")
    pools = {}
    for row in rows:
        if (int(row["clasp_id"]) != int(reference["clasp_id"])
                or normalise_text(row["context"]) != context_text or row["targets"] != targets):
            raise ValueError(f"{job_id}: inconsistent raw context/targets")
        if row["strategy"] in pools:
            raise ValueError(f"{job_id}: duplicate strategy {row['strategy']}")
        pools[row["strategy"]] = row["raws"]
    missing = set(args.strategies) - set(pools)
    if missing:
        raise ValueError(f"{job_id}: missing strategies {sorted(missing)}")
    for target in targets:
        row = ratings[target["language"]]
        if (normalise_text(row["Pre-Context"]) != context_text
                or normalise_text(row["Sentence"]) != normalise_text(target["text"])):
            raise ValueError(f"{job_id}: generation text differs from ratings CSV")
    return job_id, context_text, targets, pools


def gold_document(rows, ratings, gold, parser, args, annotator=None, nlp=None):
    """Gold context and the five target items for one CLASP ID.

    With --target-annotation corpipe the target sentences are annotated by the
    same Stanza + seeded-CorPipe pass as the alternatives, so both sides of
    every distance carry the same kind of tokens, syntax and entities."""
    job_id, context_text, targets, pools = check_rows(rows, ratings, args)
    clasp_id = int(rows[0]["clasp_id"])
    contexts, gold_targets = gold
    if bool(context_text) != bool(contexts.get(clasp_id)):
        raise ValueError(f"{job_id}: context text and gold context annotation disagree about being empty")
    context, context_aligned = ([], True) if not context_text else \
        gold_context(clasp_id, context_text, contexts[clasp_id], parser)
    if args.context_annotation == "corpipe" and context:
        context = corpipe_context(context, annotator, job_id)
    items, target_aligned = [], {}
    for target in targets:
        language = target["language"]
        if not gold_targets.get((clasp_id, language)):
            raise ValueError(f"{job_id}: no gold annotation for the {language} target")
        text = normalise_text(target["text"])
        sentence, target_aligned[language] = gold_target(
            clasp_id, language, text, gold_targets[(clasp_id, language)], parser)
        row = ratings[language]
        if args.target_annotation == "corpipe":
            annotated = annotate_target(text, context, annotator, parser, nlp, args,
                                        f"{job_id}_{language}")
            sentences, source = annotated["sentences"], annotated["annotation_source"]
        else:
            sentences, source = [sentence], "manual_gum_csv_entities_stanza_ud_on_gold_tokens"
        branch = {"id": f"target_{language}", "language": language,
                  "kind": "observed_target" if language == "English" else "back_translation_target",
                  "text": text, "sentences": sentences,
                  "annotation_source": source,
                  "ratings_with_context": ratings_list(row.get("With-Context Ratings", "")),
                  "ratings_without_context": ratings_list(row.get("Without-Context Ratings", "")),
                  "post_context": row.get("Post-Context", "")}
        items.append({"id": f"{job_id}::{language}", "language": language,
                      "sentence_index": len(context), "prefix_sentence_count": len(context),
                      "target": branch})
    return {"context_text": context_text,
            "text_alignment": {"context": context_aligned, "targets": target_aligned},
            "context": context, "items": items}, pools


# ------------------------------------------------------- paths and jobs

def raw_paths(args):
    paths = {int(p.name.split("_")[1]): p for p in args.raw_dir.glob("clasp_*_raw.jsonl")}
    if args.clasp_ids:
        if set(args.clasp_ids) - set(paths):
            raise SystemExit(f"no raw file for CLASP IDs {sorted(set(args.clasp_ids) - set(paths))}")
        paths = {i: paths[i] for i in args.clasp_ids}
    if not paths:
        raise SystemExit("no CLASP raw files found")
    return dict(sorted(paths.items()))


def settings(args):
    return {"selection": "iv_reference_first_spacy_sentence"
                         + ("; complete sentences only" if args.require_complete_sentence else "")
                         + ("; no hold-back" if args.no_hold_back else "; html/non-english held back")
                         + ("; lowercased" if args.lowercase_alternatives else ""),
            "n": args.n,
            "max_tokens": args.max_tokens, "spacy_model": args.spacy_model, "model": args.model,
            "segment": args.segment, "stanza_package": args.stanza_package,
            "context_annotation": args.context_annotation,
            "target_annotation": args.target_annotation,
            "require_complete_sentence": args.require_complete_sentence,
            "incomplete_fallback": args.incomplete_fallback,
            "pad_to_n": args.pad_to_n,
            "hold_back": args.hold_back,
            "no_hold_back": args.no_hold_back,
            "lowercase_alternatives": args.lowercase_alternatives}


def final_path(args, clasp_id):
    return args.out_dir / "clasp" / f"clasp_{clasp_id}.json"


def parts_dir(args, clasp_id):
    return args.out_dir / "clasp" / "parts" / f"clasp_{clasp_id}"


def part_path(args, clasp_id, strategy):
    return parts_dir(args, clasp_id) / f"{strategy}.{settings_hash(settings(args))}.json"


def gold_path(args, clasp_id):
    return parts_dir(args, clasp_id) / f"_gold.{settings_hash(settings(args))}.json"


def list_jobs(args):
    """One job per ID and strategy; longest gold context first."""
    contexts = load_gold(args.clasp_annotation_dir)[0]
    jobs = []
    for clasp_id in raw_paths(args):
        cost = sum(len(rows) for rows in contexts.get(clasp_id, {}).values())
        for number, strategy in enumerate(args.strategies):
            jobs.append({"key": f"clasp{clasp_id}_{strategy}", "doc": clasp_id, "strategy": strategy,
                         "cost": cost, "order": number})
    return sorted(jobs, key=lambda job: (-job["cost"], job["doc"], job["order"]))


def job_done(args, job):
    return final_path(args, job["doc"]).exists() or part_path(args, job["doc"], job["strategy"]).exists()


def clear_outputs(args):
    for clasp_id in raw_paths(args):
        final_path(args, clasp_id).unlink(missing_ok=True)
        shutil.rmtree(parts_dir(args, clasp_id), ignore_errors=True)


# --------------------------------------------------------------- worker

class Worker:
    def __init__(self, args, annotator, parser, nlp, queue):
        self.args, self.annotator, self.parser, self.nlp, self.queue = args, annotator, parser, nlp, queue
        self.paths = raw_paths(args)
        self.ratings = load_ratings(args.clasp_gold)
        self.gold = load_gold(args.clasp_annotation_dir)
        self.documents = LastUsed(3)

    def load(self, clasp_id):
        rows = list(load_jsonl(self.paths[clasp_id]))
        if not rows or int(rows[0]["clasp_id"]) != clasp_id:
            raise ValueError(f"{self.paths[clasp_id]}: empty file or wrong CLASP ID")
        document, pools = gold_document(rows, self.ratings[clasp_id], self.gold, self.parser,
                                        self.args, self.annotator, self.nlp)
        if not gold_path(self.args, clasp_id).exists():
            atomic_json(gold_path(self.args, clasp_id), {"settings": settings(self.args), **document})
        # The annotation cache lives with the ID, so repeated texts across
        # strategies handled by this worker are annotated once.
        return {"context": document["context"], "pools": pools,
                "cache": AnnotationCache(not self.args.no_annotation_cache)}

    def run(self, job):
        args, clasp_id, strategy = self.args, job["doc"], job["strategy"]
        state = self.documents.get_or_make(clasp_id, lambda: self.load(clasp_id))
        job_id = f"clasp_{clasp_id}"
        kept, qa = annotate_pool(state["pools"][strategy], strategy, state["context"], self.annotator,
                                 self.parser, self.nlp, state["cache"], args, job_id)
        print(summary_line(job_id, strategy, qa, args.n), flush=True)
        atomic_json(part_path(args, clasp_id, strategy),
                    {"settings": settings(args), "strategy": strategy, "short": len(kept) < args.n,
                     "alternatives": kept, "qa": qa})
        assemble_clasp(args, clasp_id, self.queue)


def assemble_clasp(args, clasp_id, queue=None):
    """Write the ID file once every strategy is done. Returns a status."""
    final = final_path(args, clasp_id)
    if final.exists():
        return "done"
    missing = [s for s in args.strategies if not part_path(args, clasp_id, s).exists()]
    if not gold_path(args, clasp_id).exists():
        return "not started"
    if missing:
        return f"{len(missing)}/{len(args.strategies)} strategies to do"
    if queue is not None and not queue.claim(f"assemble_clasp{clasp_id}"):
        return "being assembled by another worker"
    gold = read_json(gold_path(args, clasp_id))
    parts = [read_json(part_path(args, clasp_id, s)) for s in args.strategies]
    short = [part["strategy"] for part in parts if part["short"]]
    document = {"schema_version": 11, "complete": not short, "shortfall_strategies": short,
                "dataset": "clasp", "context_id": f"clasp_{clasp_id}", "model": args.model,
                "corpipe_segment": args.segment, "n_per_strategy": args.n,
                "strategies": list(args.strategies),
                "settings": settings(args), "context_text": gold["context_text"],
                "text_alignment": gold["text_alignment"], "context": gold["context"],
                "items": gold["items"],
                "alternatives": [a for part in parts for a in part["alternatives"]],
                "input_qa": {part["strategy"]: part["qa"] for part in parts}}
    atomic_json(final, document)
    print(f"wrote {final}" + ("" if not short else f" (INCOMPLETE: {short})"), flush=True)
    if not short and not args.keep_parts:
        shutil.rmtree(parts_dir(args, clasp_id), ignore_errors=True)
    return "done" if not short else f"incomplete {short}"


def assemble_all(args):
    return {f"clasp_{clasp_id}": assemble_clasp(args, clasp_id) for clasp_id in raw_paths(args)}


# ---------------------------------------------------------- entry points

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, dataset="clasp")
    parser.add_argument("--clasp-gold", type=Path, required=True,
                        help="processed_ratings.csv (texts and ratings)")
    parser.add_argument("--clasp-annotation-dir", type=Path,
                        help="folder with contexts_tokens.csv and processing_tokens.csv")
    parser.add_argument("--clasp-ids", type=int, nargs="+")
    return parser


# The CLASP pipeline that produced the released information-value files:
# CorPipe annotates the context, the target and every alternative, so all three
# share one mention inventory and one set of cluster ids.
FIXED = ["--context-annotation", "corpipe", "--target-annotation", "corpipe",
         "--lowercase-alternatives"]


def _with_fixed(argv):
    argv = list(argv)
    return [a for a in FIXED if a not in argv] + argv


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(_with_fixed(sys.argv[1:] if argv is None else argv))
    check_args(parser, args)
    if not args.clasp_annotation_dir:
        parser.error("--clasp-annotation-dir is required")
    return args


def main():
    args = parse_args()
    raw_paths(args)
    if args.overwrite:
        clear_outputs(args)
    queue = WorkQueue(args.claim_dir, args.worker_name)
    jobs = list_jobs(args)
    if all(job_done(args, job) for job in jobs):
        print("nothing to do", flush=True)
    else:
        annotator, parser, nlp = initialise(args)
        worker = Worker(args, annotator, parser, nlp, queue)
        _, failed = run_jobs(jobs, lambda job: job_done(args, job), worker.run, queue)
        if args.claim_dir:  # run_parallel.py assembles and reports at the end
            sys.exit(1 if failed else 0)
    report = assemble_all(args)
    for name, status in report.items():
        print(f"{name}: {status}", flush=True)
    if any(status != "done" for status in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
