# Entity Information Value

Information Value is the expected distance between the utterance that actually
occurred and plausible alternative  utterances. 
This directory holds the four stages that produce it, plus the
gold entity counts used alongside it.

    1. generate    sample alternative continuations with vLLM
    2. annotate    parse and coreference-annotate them against a frozen context
    3. iv          score each target against its alternatives
    4. counts      count entities in the gold annotation (no alternatives)

Stages 3 and 4 are independent: the counts need only the gold annotation.

## Setup

    uv sync                        # stages 2, 3, 4
    uv sync --extra generation     # adds vLLM, needed only for stage 1

`en_core_web_sm` installs automatically as a dependency. Everything below is
run with `uv run python <script>` from this directory.

## The coreference model is not vendored

Stage 2 runs CorPipe, which is **not** included here. `corpipe26_seeded.py`
loads an upstream `corpipe26_twostage.py` at run time and drives it with a
frozen context, so the alternatives are annotated against exactly the same
preceding discourse as the target and cluster ids are shared between them.

Get CorPipe from <https://github.com/ufal/crac2026-corpipe> and pass the file
path with `--corpipe-source`. It brings its own requirements, `minnt` among
them; without it you will see `ModuleNotFoundError: No module named 'minnt'`.
The model weights are pulled from the Hugging Face hub by name, the default
being `ufal/corpipe26-twostage-corefud1.4-large-260702`.

## 1. Generate alternatives

500 continuations per item under 11 decoding strategies (ancestral, three
temperatures, four nucleus, four typical). Needs GPUs.

    uv run python generate_ns_vllm.py \
        --csv-dir      <manual story CSVs> \
        --model        <path to gemma-2-9b> \
        --output-dir   raw_ns_vllm \
        --gpus 4 --samples 500 --batch-size 128 --max-new-tokens 120

    uv run python generate_clasp_vllm.py \
        --clasp-data   processed_ratings.csv \
        --model        <path to gemma-2-9b> \
        --output-dir   raw_clasp_vllm \
        --gpus 4 --raws-per-strategy 500 --batch-size 128 --max-new-tokens 120

`vllm_common.py` and `typical_sampling.py` are support modules for these two.

## 2. Annotate the alternatives

Each raw sample is filtered to a complete first sentence, parsed with Stanza and
coreference-annotated by seeded CorPipe against the frozen gold context. Of the
500 raws per item, `--n` are kept.

Two pipelines are shipped, one per dataset, because they are the two that
produced the released information-value files:

| script | dataset | context | target and alternatives | syntax |
|---|---|---|---|---|
| `annotate_ns.py` | Natural Stories | manual story CSVs | CorPipe | Stanza |
| `annotate_clasp.py` | CLASP | CorPipe | CorPipe | Stanza |

Natural Stories keeps the manual annotation for the context, so cluster
identity is anchored to the gold data, while target and alternatives are
annotated by the same pass as each other. On CLASP everything is CorPipe.

`run_parallel.py` shards the work over GPUs and resumes: rerunning the same
command skips items that already have a part file.

    uv run python run_parallel.py ns \
        --workers-per-gpu 2 --cpu-threads 4 -- \
        --raw-dir raw_ns_vllm \
        --entity-dir <manual story CSVs> \
        --corpipe-source <path to corpipe26_twostage.py> \
        --stanza-gpu --n 90 --out-dir annotated_alternatives

    uv run python run_parallel.py clasp \
        --workers-per-gpu 2 --cpu-threads 4 -- \
        --raw-dir raw_clasp_vllm \
        --clasp-gold processed_ratings.csv \
        --clasp-annotation-dir <manual CLASP token CSVs> \
        --corpipe-source <path to corpipe26_twostage.py> \
        --stanza-gpu --n 90 --out-dir annotated_alternatives

Both write into the same output directory but touch different subfolders, so
they can run at the same time. `run_parallel.py` merges the per-item part files
into one document per story or context at the end; if it is interrupted at that
point the part files are intact and rerunning the command completes the merge.

## What each file is

| file | what it does |
|---|---|
| `generate_ns_vllm.py`, `generate_clasp_vllm.py` | stage 1, sample alternatives |
| `vllm_common.py`, `typical_sampling.py` | support for the two generators |
| `annotate_ns.py`, `annotate_clasp.py` | stage 2, one per dataset: filter the raw samples, parse them, annotate coreference |
| `annotate_common.py` | the filtering and annotation logic both share |
| `run_parallel.py` | shards stage 2 over GPUs and resumes |
| `corpipe26_seeded.py` | drives CorPipe against a frozen context |
| `textproc.py` | shared text handling |
| `calculate_entity_iv.py` | stage 3, the 45 distance columns |
| `entity_roles.py` | the one definition of a mention's grammatical role |
| `build_gum_transition_cost.py` | fits the role-transition model from GUM |
| `gum_deprel_costs_50.json` | that fitted model |
| `mention_role_costs.py` | stage 4a, cost of each mention's role transition |
| `count_entities_ns.py`, `count_entities_clasp.py` | stage 4b, accumulate those per item |
| `read_ns_annotation.py` | reads the manual story CSVs and their entity brackets |

There is one annotation script per dataset rather than a generic one plus a
configuration wrapper. The choices that distinguish the pipelines are fixed
inside each script, in its `FIXED` list, and passing them again on the command
line is refused: on Natural Stories the target and the alternatives must be
annotated by the same CorPipe pass over Stanza syntax, and on CLASP the context
is annotated by CorPipe too. A target annotated differently from its
alternatives would make the comparison between them meaningless, so these are
not options.

## 3. Calculate Information Value

Reads the annotated documents and scores every target against its alternatives
on 15 distance functions, each aggregated three ways (mean, 80th percentile,
minimum), giving 45 columns per strategy.

    uv run python calculate_entity_iv.py \
        annotated_alternatives/naturalstories annotated_alternatives/clasp \
        --transition-costs gum_deprel_costs_50.json \
        --role-inventory deprel \
        --samples-per-strategy 90 \
        --out-dir iv_values

Add `--strategies temp_125` to restrict to one strategy and `--device cpu` to
run without a GPU (only the sentence embeddings use one).

`entity_roles.py` is the single definition of a mention's grammatical role,
shared by the information-value and counting scripts so the two cannot drift.

### The role-transition cost model

`gum_deprel_costs_50.json` is the fitted model that `d3_transition` scores
against: a 30x30 matrix of `-log2 P(role | the same entity's previous role)`,
in bits.

It is fitted by `build_gum_transition_cost.py` from the GUM release. Download
GUM from <https://github.com/amir-zeldes/gum> and point the script at the
archive or directory of `dep/*.conllu` files:

    uv run python build_gum_transition_cost.py \
        --gum <GUM dep/ conllu archive or directory> \
        --role-inventory deprel --min-label-count 50 --smoothing 0.5 \
        --output gum_deprel_costs_50.json

Every mention in an entity chain is paired with that entity's immediately
preceding mention in the document, even across intervening sentences that do
not mention it. A mention's role is the Stanza dependency label of its span
head, with `conj` and `appos` inheriting the role of what they attach to. Only
labels seen at least `--min-label-count` times become states; rarer ones fold
into `other`. The shipped model was fitted on 275 documents, 39,816 entity
chains, 76,967 mentions and 37,151 transitions.

The JSON stores the raw contingency table as well as the costs, under `counts`
and `row_totals`, so the fit can be checked without GUM:

    cost(prev -> cur) = -log2((counts[prev][cur] + smoothing)
                              / (row_totals[prev] + smoothing * n_states))

All 900 cells of the shipped file reproduce exactly under that formula with
`smoothing = 0.5` and `n_states = 30`.

`--role-inventory` also accepts `grid` (3 states), `coarse` (4) and `fine`
(12). Whatever is used here must match `--role-inventory` in stage 3 and stage
4; `calculate_entity_iv.py` refuses a cost file fitted with a different
inventory rather than silently mixing them.

## 4. Gold entity counts

These use the gold annotation only. No alternatives and no language model.

**Step one**, score each mention's role transition against the GUM model:

    # Natural Stories, from the manual annotation
    uv run python mention_role_costs.py \
        --ns-csv-dir <manual story CSVs> \
        --role-costs gum_deprel_costs_50.json --role-inventory deprel \
        --out-dir mention_costs

    # CLASP, from the annotation produced in stage 2
    uv run python mention_role_costs.py \
        --clasp-json-dir annotated_alternatives/clasp \
        --role-costs gum_deprel_costs_50.json --role-inventory deprel \
        --out-dir mention_costs

The CLASP source matters. The counts must come from the same annotation as the
CLASP information-value files, which is the stage 2 output, not the manual
token CSVs: the two disagree on the mention count for roughly two thirds of
targets.

**Step two**, accumulate them per item:

    uv run python count_entities_ns.py \
        --mention-costs mention_costs/naturalstories/mention_costs.csv \
        --entity-dir <manual story CSVs> \
        --out naturalstories_entity_counts.csv

    uv run python count_entities_clasp.py \
        --mention-costs mention_costs/clasp/mention_costs.csv \
        --annotation-dir annotated_alternatives/clasp \
        --out count_entities_clasp.csv

`--entity-dir` and `--annotation-dir` are used only to enumerate items, so that
an item in which the annotator found no mentions still gets a row of zeros
rather than disappearing. Three Natural Stories sentences and six CLASP targets
are like this.

`read_ns_annotation.py` reads the manual story CSVs and parses their GUM-style
entity brackets.

## Joining the outputs

`sentence_index` in the information-value files and `sent_id` in the Natural
Stories counts are both **0-based**, matching the manual annotation CSVs.
Natural Stories reading-time data numbers sentences from 1, so that join needs
`+ 1`. On CLASP the unit is `item_id`, written `clasp_<id>::<Language>`, because
the five language versions of a context are separate targets sharing one
context.

## Verifying an install

Stage 4 needs no GPU and runs in under a minute, so it is the quickest check
that the environment is correct. Run both steps above against a known
annotation and compare the CSVs with a previous run; they are deterministic and
should match exactly.

Stage 3 is deterministic too, except for the two `d_semantic_*` channels, which
differ by around 1e-7 between CPU and GPU because the embeddings are computed in
floating point on different hardware. The other 39 of the 45 columns are
bit-identical across devices.
