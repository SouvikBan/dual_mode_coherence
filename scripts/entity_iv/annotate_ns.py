#!/usr/bin/env python3
"""Natural Stories: gold story plus N annotated alternatives per sentence and strategy.

Keep beside annotate_common.py and corpipe26_seeded.py.

Tokens and entities come from the annotated story CSVs. Syntax (--ns-syntax):
  stanza   Stanza UD on the gold tokens, the same parser as the alternatives
           (default: target and alternatives must carry one annotation
           standard, or the difference between the two parsers is measured as
           information value);
  gold-v2  the CSV trees relabelled from UD v1 to UD v2 (the previous default;
           the CSVs use v1 labels such as neg, mwe, dobj, and nmod for
           obliques, and the IV script rejects neg and mwe);
  gold     the CSV labels unchanged (fails in the IV script).
One job = one story sentence with all strategies. Finished jobs are saved as
part files (OUT/naturalstories/parts/story_N/), so an interrupted run resumes
and several workers can share the stories (see run_parallel.py). The story
file is assembled when all its sentences are done.
"""

import argparse
import copy
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from annotate_common import (
    AnnotationCache, LastUsed, WorkQueue, add_common_args, annotate_pool, annotate_target,
    corpipe_context,
    atomic_json, build_gold_sentence, check_args, check_iv_labels, convert_ud1_to_ud2,
    entity_spans, initialise, read_json, run_jobs, settings_hash, summary_line,
)

CSV_COLUMNS = ["story_id", "sent_id", "token_id", "deprel", "form", "head", "pos", "entity_layer"]
# Lines written by the generator start with sent_id and strategy.
RAW_PREFIX_RE = re.compile(rb'^\{"sent_id": (\d+), "strategy": "([^"]+)"')


def story_number(path):
    match = re.search(r"story_(\d+)_", Path(path).name)
    if not match:
        raise ValueError(f"cannot read story number from {path}")
    return int(match.group(1))


def read_story_rows(path, story_id):
    sentences = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if set(reader.fieldnames or []) != set(CSV_COLUMNS):
            raise ValueError(f"{path}: expected columns {CSV_COLUMNS}")
        for row in reader:
            if int(row["story_id"]) != story_id:
                raise ValueError(f"{path}: mixed story IDs")
            sentences[int(row["sent_id"])].append(row)
    ids = sorted(sentences)
    if ids != list(range(len(ids))):
        raise ValueError(f"{path}: sentence IDs must be contiguous from zero")
    # File order is token order. Sorting token_id strings breaks on ids such
    # as 322.word / 322.4 (story 10) and gives cyclic trees.
    return [sentences[i] for i in ids]


# ------------------------------------------- natural (untokenised) text
#
# The gold CSV forms are Penn Treebank tokens, so " ".join(forms) writes
# "England , you" and PTB quotes ``/''. Alternatives are free model text. The
# reading-time presentation file all_stories.tok holds the same stories as the
# readers saw them, and the gold forms align to it exactly (10/10 stories, one
# spelling difference: gold "peeked" vs presentation "peaked" in story 2).

PTB_FORMS = {"``": ['"', "\u201c", "'", "\u2018"], "''": ['"', "\u201d", "'", "\u2019"],
             "-LRB-": ["("], "-RRB-": [")"], "-LCB-": ["{"], "-RCB-": ["}"],
             "-LSB-": ["["], "-RSB-": ["]"], "--": ["\u2014", "\u2013", "--"]}


def form_variants(form):
    if form in PTB_FORMS:
        return PTB_FORMS[form]
    return list(dict.fromkeys([form, form.replace("\u2019", "'"), form.replace("'", "\u2019"),
                               form.replace("\u201c", '"').replace("\u201d", '"')]))


def read_presentation_stories(path):
    """story number -> the story as one string of space-separated zones."""
    stories = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            stories[int(row["item"])].append(row["word"])
    return {story_id: " ".join(words) for story_id, words in stories.items()}


def align_story_text(text, rows_by_sentence, story_id):
    """Per sentence: (text, [(char_start, char_end)]) inside the natural text."""
    cursor, output = 0, []
    for index, rows in enumerate(rows_by_sentence):
        offsets = []
        for row in rows:
            form = row["form"]
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            length = None
            for variant in form_variants(form):
                if text.startswith(variant, cursor):
                    length = len(variant)
                    break
            if length is None:  # tolerate a one-character spelling difference
                candidate = text[cursor:cursor + len(form)]
                if (len(candidate) == len(form)
                        and sum(a != b for a, b in zip(candidate.lower(), form.lower())) <= 1):
                    length = len(form)
            if length is None:
                raise ValueError(
                    f"story {story_id} sentence {index}: cannot align gold form {form!r} "
                    f"to the presentation text at {text[cursor:cursor + 40]!r}")
            offsets.append((cursor, cursor + length))
            cursor += length
        base = offsets[0][0]
        output.append((text[base:offsets[-1][1]], [(a - base, b - base) for a, b in offsets]))
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor != len(text):
        raise ValueError(f"story {story_id}: {len(text) - cursor} characters of the "
                         f"presentation text were not covered by the gold tokens")
    return output


def build_story(rows_by_sentence, story_id, syntax, parser, natural=None):
    """natural: [(text, offsets)] from align_story_text, or None for
    " ".join(gold forms) (Penn Treebank spacing and quotes)."""
    forms = [[row["form"] for row in rows] for rows in rows_by_sentence]
    parsed = parser.parse_pretokenized(forms) if syntax == "stanza" else None
    label = {"stanza": "stanza_ud_on_gold_tokens", "gold-v2": "naturalstories_csv_ud1_to_ud2",
             "gold": "naturalstories_csv"}[syntax]
    story = []
    for index, rows in enumerate(rows_by_sentence):
        gold_syntax = None if parsed else [(int(r["head"]), r["deprel"], r["pos"]) for r in rows]
        text, offsets = natural[index] if natural else (None, None)
        sentence = build_gold_sentence(
            forms[index], text, offsets, entity_spans([row["entity_layer"] for row in rows]),
            words_by_token=parsed[index] if parsed else None, gold_syntax=gold_syntax,
            meta={"sentence_index": index, "source_story_id": f"naturalstories_{story_id}",
                  "source_sentence_index": index},
            syntax_label=label, token_ids=[row["token_id"] for row in rows])
        if syntax == "gold-v2":
            convert_ud1_to_ud2(sentence["tokens"])
        if syntax != "gold":
            check_iv_labels(sentence, f"story {story_id} sentence {index}")
        story.append(sentence)
    return story


def observed_target(sentence):
    target = copy.deepcopy(sentence)
    target["sentence_index"] = 0
    return {"id": "target", "kind": "observed_target", "text": sentence["text"],
            "sentences": [target], "annotation_source": "manual_csv_tokens_and_entities"}


# ------------------------------------------------------------- raw pools

def index_raw_file(path, n_sentences, strategies, story_id):
    """Byte offset of every (sent_id, strategy) line, so a worker reads only
    the lines of the sentence it works on (a story file can be ~0.5 GB)."""
    index, offset = {}, 0
    with Path(path).open("rb") as stream:
        for line in stream:
            if line.strip():
                match = RAW_PREFIX_RE.match(line)
                if match:
                    key = int(match.group(1)), match.group(2).decode()
                else:
                    row = json.loads(line)
                    key = int(row["sent_id"]), row["strategy"]
                if key in index:
                    raise ValueError(f"story {story_id}: duplicate raw pool {key}")
                if not 0 <= key[0] < n_sentences:
                    raise ValueError(f"story {story_id}: unknown sentence {key[0]}")
                index[key] = (offset, len(line))
            offset += len(line)
    missing = {(s, t) for s in range(n_sentences) for t in strategies} - index.keys()
    if missing:
        raise ValueError(f"story {story_id}: missing raw pools {sorted(missing)[:5]}")
    return index


def read_pool(path, index, sent_id, strategy):
    offset, length = index[(sent_id, strategy)]
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        row = json.loads(stream.read(length))
    if int(row["sent_id"]) != sent_id or row["strategy"] != strategy:
        raise ValueError(f"raw index mismatch at {(sent_id, strategy)}")
    return row["raws"]


# ------------------------------------------------------- paths and jobs

def story_paths(args):
    paths = sorted(args.entity_dir.glob("story_*_entity_annotated.csv"), key=story_number)
    if args.stories:
        paths = [p for p in paths if story_number(p) in args.stories]
        if {story_number(p) for p in paths} != set(args.stories):
            raise SystemExit("some requested stories have no entity CSV")
    if not paths:
        raise SystemExit("no annotated story CSVs found")
    if len({story_number(p) for p in paths}) != len(paths):
        raise SystemExit("multiple CSVs have the same story ID")
    return {story_number(p): p for p in paths}


def settings(args):
    return {"selection": "iv_reference_first_spacy_sentence"
                         + ("; complete sentences only" if args.require_complete_sentence else "")
                         + ("; no hold-back" if args.no_hold_back else "; html/non-english held back"),
            "n": args.n,
            "strategies": list(args.strategies), "syntax": args.ns_syntax, "max_tokens": args.max_tokens,
            "spacy_model": args.spacy_model, "model": args.model, "segment": args.segment,
            "stanza_package": args.stanza_package, "ns_text": args.ns_text,
            "context_annotation": args.context_annotation,
            "target_annotation": args.target_annotation,
            "require_complete_sentence": args.require_complete_sentence,
            "incomplete_fallback": args.incomplete_fallback,
            "pad_to_n": args.pad_to_n,
            "hold_back": args.hold_back,
            "no_hold_back": args.no_hold_back}


def final_path(args, story_id):
    return args.out_dir / "naturalstories" / f"naturalstories_{story_id}.json"


def parts_dir(args, story_id):
    return args.out_dir / "naturalstories" / "parts" / f"story_{story_id}"


def part_path(args, story_id, sent_id):
    return parts_dir(args, story_id) / f"s{sent_id:03d}.{settings_hash(settings(args))}.json"


def gold_path(args, story_id):
    return parts_dir(args, story_id) / f"_story.{settings_hash(settings(args))}.json"


def list_jobs(args):
    """One job per sentence; longest context first (the slowest jobs)."""
    jobs = []
    for story_id, path in story_paths(args).items():
        if not (args.raw_dir / f"story_{story_id}_raw.jsonl").exists():
            raise SystemExit(f"missing raw file for story {story_id}")
        context_tokens = 0
        for sent_id, rows in enumerate(read_story_rows(path, story_id)):
            jobs.append({"key": f"story{story_id}_s{sent_id:03d}", "doc": story_id,
                         "sent_id": sent_id, "cost": context_tokens})
            context_tokens += len(rows)
    return sorted(jobs, key=lambda job: (-job["cost"], job["key"]))


def job_done(args, job):
    return final_path(args, job["doc"]).exists() or part_path(args, job["doc"], job["sent_id"]).exists()


def clear_outputs(args):
    for story_id in story_paths(args):
        final_path(args, story_id).unlink(missing_ok=True)
        shutil.rmtree(parts_dir(args, story_id), ignore_errors=True)


# --------------------------------------------------------------- worker

class Worker:
    def __init__(self, args, annotator, parser, nlp, queue):
        self.args, self.annotator, self.parser, self.nlp, self.queue = args, annotator, parser, nlp, queue
        self.paths = story_paths(args)
        self.stories = LastUsed(2)

    def load_story(self, story_id):
        args = self.args
        rows = read_story_rows(self.paths[story_id], story_id)
        natural = None
        if args.ns_text == "natural":
            text = read_presentation_stories(args.stories_tok)[story_id]
            natural = align_story_text(text, rows, story_id)
        story = build_story(rows, story_id, args.ns_syntax, self.parser, natural)
        if args.context_annotation == "corpipe":
            story = corpipe_context(story, self.annotator, f"naturalstories_{story_id}")
        if not gold_path(args, story_id).exists():
            atomic_json(gold_path(args, story_id), {"settings": settings(args), "context": story})
        raw = args.raw_dir / f"story_{story_id}_raw.jsonl"
        return story, raw, index_raw_file(raw, len(rows), args.strategies, story_id)

    def run(self, job):
        args, story_id, sent_id = self.args, job["doc"], job["sent_id"]
        story, raw, index = self.stories.get_or_make(story_id, lambda: self.load_story(story_id))
        job_id = f"naturalstories_{story_id}::s{sent_id}"
        cache = AnnotationCache(not args.no_annotation_cache)
        alternatives, qa, short = [], {}, []
        for strategy in args.strategies:
            kept, qa[strategy] = annotate_pool(read_pool(raw, index, sent_id, strategy), strategy,
                                               story[:sent_id], self.annotator, self.parser, self.nlp,
                                               cache, args, job_id)
            alternatives.extend(kept)
            if len(kept) < args.n:
                short.append(strategy)
            print(summary_line(job_id, strategy, qa[strategy], args.n), flush=True)
        if args.target_annotation == "corpipe":
            target = annotate_target(story[sent_id]["text"], story[:sent_id], self.annotator,
                                     self.parser, self.nlp, args, job_id, cache)
        else:
            target = observed_target(story[sent_id])
        item = {"id": job_id, "sentence_index": sent_id, "prefix_sentence_count": sent_id,
                "target": target, "alternatives": alternatives, "input_qa": qa}
        atomic_json(part_path(args, story_id, sent_id), {"settings": settings(args), "short": short, "item": item})
        assemble_story(args, story_id, self.queue)


def assemble_story(args, story_id, queue=None):
    """Write the story file once every sentence is done. Returns a status."""
    final = final_path(args, story_id)
    if final.exists():
        return "done"
    if not gold_path(args, story_id).exists():
        return "not started"
    gold = read_json(gold_path(args, story_id))
    n_sentences = len(gold["context"])
    missing = [k for k in range(n_sentences) if not part_path(args, story_id, k).exists()]
    if missing:
        return f"{len(missing)}/{n_sentences} sentences to do"
    if queue is not None and not queue.claim(f"assemble_story{story_id}"):
        return "being assembled by another worker"
    parts = [read_json(part_path(args, story_id, k)) for k in range(n_sentences)]
    shortfalls = [f"s{k}/{strategy}" for k, part in enumerate(parts) for strategy in part["short"]]
    document = {"schema_version": 11, "complete": not shortfalls, "shortfalls": shortfalls,
                "dataset": "naturalstories", "story_id": f"naturalstories_{story_id}",
                "model": args.model, "corpipe_segment": args.segment, "n_per_strategy": args.n,
                "strategies": list(args.strategies), "syntax": args.ns_syntax,
                "settings": settings(args), "context": gold["context"],
                "items": [part["item"] for part in parts]}
    atomic_json(final, document)
    print(f"wrote {final}" + ("" if not shortfalls else f" (INCOMPLETE: {shortfalls[:5]})"), flush=True)
    if not shortfalls and not args.keep_parts:
        shutil.rmtree(parts_dir(args, story_id), ignore_errors=True)
    return "done" if not shortfalls else f"incomplete {shortfalls[:5]}"


def assemble_all(args):
    return {f"naturalstories_{story_id}": assemble_story(args, story_id) for story_id in story_paths(args)}


# ---------------------------------------------------------- entry points

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, dataset="naturalstories")
    parser.add_argument("--entity-dir", type=Path, required=True)
    parser.add_argument("--stories", type=int, nargs="+")
    parser.add_argument("--ns-syntax", choices=("stanza", "gold-v2", "gold"), default="stanza",
                        help="stanza (default): parse the gold tokens with the same Stanza model "
                             "used for the alternatives, so target and alternatives carry one "
                             "annotation standard; gold-v2: the CSV trees (UD v1 from the CoreNLP "
                             "converter) relabelled to UD v2 - comparable in label inventory but "
                             "not in analysis (90.3%% label agreement, 81.3%% UAS, 3-gram distance "
                             "0.236 against Stanza on the same sentence); gold: labels unchanged "
                             "(the IV script rejects neg/mwe)")
    parser.add_argument("--ns-text", choices=("tokens", "natural"), default="tokens",
                        help="tokens: \" \".join(gold Penn Treebank forms) (default, as before); "
                             "natural: the story as the readers saw it, from --stories-tok")
    parser.add_argument("--stories-tok", type=Path,
                        default=Path("../naturalstories/naturalstories_RTS/all_stories.tok"),
                        help="reading-time presentation file, used by --ns-text natural")
    return parser


# The pipeline this repository ships is the one that produced the released
# Natural Stories information-value files: the manual story CSVs supply the
# context, while the target sentence and every alternative are annotated by the
# same CorPipe pass, over Stanza syntax throughout. Those two choices are fixed
# rather than optional, because a target annotated differently from its
# alternatives would make the comparison between them meaningless.
FIXED = ["--target-annotation", "corpipe", "--ns-syntax", "stanza"]


def _with_fixed(argv):
    argv = list(argv)
    for flag in ("--target-annotation", "--ns-syntax", "--syntax"):
        if flag in argv:
            raise SystemExit(f"{flag} is fixed: the target and the alternatives "
                             "must be annotated by the same pass")
    return FIXED + argv


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(_with_fixed(sys.argv[1:] if argv is None else argv))
    check_args(parser, args)
    if args.ns_text == "natural" and not args.stories_tok.is_file():
        parser.error(f"--ns-text natural needs the presentation file; {args.stories_tok} is missing")
    if args.lowercase_alternatives:
        parser.error("--lowercase-alternatives is a CLASP option; Natural Stories keeps its casing")
    return args


def main():
    args = parse_args()
    story_paths(args)
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
