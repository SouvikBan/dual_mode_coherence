# Code behind the paper

Everything needed to recompute the numbers, tables and figures of *Using Generalized Surprisal to
Model Readers' Expectations about Discourse Entities*, from the raw corpora to the appendix. The
chain has four parts; the first two live in `../scripts/`, the last two here.

```
1. alternatives     ../scripts/entity_iv/   sample 500 continuations per utterance with Gemma-2-9B,
                                            keep 90 per strategy, parse and coreference-annotate them
2. surprisal        ../scripts/             word surprisal of the targets (Pythia), CLASP rating table
3. measures         paper/measures/         the entity and non-entity measures of Section 3 and 4,
                                            one row per target and sampling strategy
4. analyses         paper/analysis/         mixed models, held-out prediction, figures, tables, appendix
```

## 1. Alternatives: generation and annotation

Done once per dataset by the scripts in `../scripts/entity_iv/` (its README has the full
commands and options).

**Inputs.** Natural Stories: the ten story texts with the manual coreference annotation
(`story_<k>_entity_annotated.csv`, GUM scheme, singletons included; `read_ns_annotation.py` parses
them) and the self-paced reading times of Futrell et al. (2021), `processed_RTs.tsv`. CLASP: the
BLL2018 release, `processed_ratings.csv` (100 Wikipedia sentences, each in its original form and four
round-trip translations, with the article text before it) and the raw MTurk ratings.

**Stage 1, sampling** (`generate_ns_vllm.py`, `generate_clasp_vllm.py`; vLLM, GPUs). For every
target the context is the story (or article) up to the previous sentence. Gemma-2-9B continues it
500 times under each of 11 strategies: ancestral sampling, temperatures 0.75 and 1.25, nucleus
p in {0.80, 0.85, 0.90, 0.95}, typical tau in {0.20, 0.30, 0.85, 0.95}. Output: one JSONL of raw
samples per story / context.

**Stage 2, annotation** (`annotate_ns.py`, `annotate_clasp.py` through `run_parallel.py`). Each raw
sample is cut to its first complete sentence, parsed with Stanza (Universal Dependencies) and
coreference-resolved by CorPipe (`corpipe26-twostage-corefud1.4-large`, driven by
`corpipe26_seeded.py`) against the frozen context, so that an alternative's mentions share cluster
ids with the context and with the target. On Natural Stories the context keeps the manual
annotation; the target and the alternatives are resolved by CorPipe. On CLASP everything is
CorPipe. The first 90 samples per strategy that parse and resolve are kept. Output: one JSON per
story / context (`annotated_alternatives/<dataset>/<document>.json`) with the context sentences,
each target and its 990 alternatives, every sentence carrying tokens (lemma, UPOS, XPOS, head,
deprel) and mentions (span, cluster id, head).

**Stage 4 of that pipeline, gold counts** (`mention_role_costs.py`, `count_entities_*.py`) gives
the number of mentions in the context before each target (`ctx_n_mentions`), which the models use
as a control. Stage 3 there (`calculate_entity_iv.py`) computes the earlier channel set of the
released files; the paper's measures are computed by `paper/measures/` instead, which only reuses
its JSON reader and the mention helpers (`entity_roles.py`, `form_inventory.py`).

## 2. Surprisal and the CLASP rating table

* Natural Stories: `../scripts/score_stories.py --model-name-or-path EleutherAI/pythia-70m-deduped
  --revision step1000` scores every word with the whole story as left context; the analyses sum
  it over the words of an utterance (`annotated/surprisal/pythia-70m-deduped-step1000-full.tsv`).
* CLASP: `../scripts/preprocess_clasp.Rmd` turns the raw MTurk file into one row per participant
  and version (`clasp_with_context_by_worker_and_sent.csv`); `../scripts/score_clasp.py
  --model-name-or-path EleutherAI/pythia-12b-deduped` adds the surprisal of each version given
  its article context (an empty context is scored after the BOS token only). The analyses use
  the mean surprisal per word. The other Pythia sizes for the surprisal-size appendix are scored
  with the same script.

## 3. Measures (`measures/`)

Each script is `python <script> <annotated json or directory> <out csv>` and writes one row per
target and strategy; `run_measures.sh` runs all of them (`measures.sbatch` on a GPU node), writing
`paper/data/measures/<measure>/<dataset>/`. `common.py` holds what they share: the record
iterator, mention extraction, the head lemma that represents an entity (its first non-pronominal
mention), a cached `all-distilroberta-v1` embedder.

| script | paper | columns |
|---|---|---|
| `established_entities.py` | claim (i), Eq. 4-5 | `d_est`, its anticipatory form, the numbers of established, mentioned-established and new entities |
| `part_of_story.py` | claim (ii), Eq. 6-7 | `d_part`, `d_part_anticipatory`; the footnote-1 variant over earlier utterances (`d_part_history*`) |
| `topic.py` + `train_salience.py` | claim (iii), Eq. 8-9 | `d_top` and its anticipatory form; the bits s and phi under the learned-salience topic and under the most-mentioned entity; `train_salience.py` fits the salience model on GUM (`gum_salience_model.json`, AUC 0.88) |
| `linking_cost.py` | claim (iv), Eq. 10-11 | `beta`, `d_beta`, `d_beta_anticipatory`, `n_new` |
| `non_entity.py` | Section 4 | k-gram distances over words, POS and dependency labels (k = 1, 2, 3) and the sentence-embedding cosine, each with its anticipatory form |
| `first_mention_zones.py`, `word_surprisal.py`, `word_zipf.py` | Appendix A, word level | zones of the target's first mention; per-word Pythia-70m surprisal; Zipf frequencies |
| `train_next_mention.py` | Appendix A, claims not reported; topic | next-mention models fitted on GUM (conditional logit per account: grammatical, recency, frequency, information status, full), antecedent-distance distributions per referring form and the accessibility distance between forms (`gum_next_mention_model.json`) |
| `next_mention.py` (Natural Stories) | Appendix A, claims not reported; Section 5 (i) | surprisal and entropy of the target's first referent under each account over the entities of the previous utterance, with the uniform `log2_M`; for targets that begin with a personal pronoun, the Expectancy, Mirror and Bayesian surprisals from the alternatives' joint table and the uniform surprisal over the agreement-compatible candidates |
| `referring_forms.py` | Appendix A, claims not reported | information value of the target's referring forms under the Givenness Hierarchy and under the typical antecedent distance, over all established entities and for the first referent |
| `topic_next_mention.py` | Appendix A, "Topic" | the topic bits under the entity the next-mention model makes most expected, and whether it is the most-mentioned entity |

## 4. Analyses (`analysis/`)

`NCORES=11 bash analysis/run_all.sh` runs everything in order (`analysis.sbatch` for a compute
node; the CLASP held-out evaluation takes hours). `paths.R` lists every external input (edit it
for another machine), `setup.R` the shared code: controls (surprisal, length, position, mentions
in the context; all predictors standardised), random intercepts (participant, story or context,
utterance or version), model fitting (lmer on summed reading times; glmmTMB ordered beta on
(rating - 1)/3), the held-out likelihoods, and the model-comparison helper of the appendix scripts.

| script | paper | output |
|---|---|---|
| `01_prepare_naturalstories.R` | Appendix A "Data": words read in 200 to 2,000 ms by participants with >= 4 correct answers, summed per utterance; 40,387 observations | `data/naturalstories_reading_times.csv` |
| `02_estimates.R <dataset>` | each measure alone and the full model, 11 strategies; CLASP on all and on the 80 contexts with established entities | `results/estimates_<dataset>.csv`, `_summary.csv` |
| `03_clasp_rating_scale.R` | logits to rating points per SD (average marginal effect of the expected rating; 0.809 per logit) | `results/clasp_rating_scale_factors.csv`, `results/estimates_clasp_rating_points*.csv` |
| `04_heldout.R <dataset>` | Appendix B, Tables 2 and 3: unseen stories / contexts and unseen participants, 95% intervals over folds, R2 and RMSE with the participant intercept | `results/heldout_<dataset>*.csv` |
| `05_figure_estimates.R` | Figures 1 and 2 | `figures/fig_estimates*.pdf/.png` |
| `06_figure_strategies.R` | Figure 3 | `figures/fig_strategies_*.pdf/.png` |
| `07_tables.py` | LaTeX bodies of Tables 1, 2, 3 | `results/tables/*.tex` |
| `appendix/position.R` | "Position", "Mentions in the context": descriptives and the nonlinear position controls | `results/appendix_position*.csv` |
| `appendix/lead_lag.R` | "Position": values at preceding and following utterances | `results/appendix_lead_lag*.csv` |
| `appendix/established_count.R` | Section 5 (ii): the number of established entities mentioned; d_part given it | `results/appendix_established_count*.csv` |
| `appendix/linking_cost.R <dataset>` | "Linking cost", Section 5 (iv) | `results/appendix_linking_cost_*.csv` |
| `appendix/word_level.R` | "Word-level localization" | `results/appendix_word_level.csv` |
| `appendix/topic.R <dataset>` | "Topic", Section 5 (iii): topic definitions, subject vs first mention | `results/appendix_topic_*.csv` |
| `appendix/next_mention.R` | "Claims tested and not reported above": next-mention accounts on the targets whose first referent comes from the previous utterance, against the uniform expectation; pronoun models on the targets that begin with a personal pronoun | `results/appendix_next_mention_*.csv` |
| `appendix/referring_forms.R <dataset>` | "Claims tested and not reported above": the distance over referring forms | `results/appendix_referring_forms_*.csv` |
| `appendix/clasp_surprisal_size.R` | "Surprisal model on CLASP" | `results/appendix_clasp_surprisal_size.csv` |
| `appendix/clasp_language_control.R` | "Language of the CLASP version" | `results/appendix_clasp_language_control*.csv` |

The paragraph "Claims tested and not reported above" is covered by `train_next_mention.py`,
`next_mention.py`, `referring_forms.py` and the two appendix scripts above; the next-mention topic
of the "Topic" paragraph by `topic_next_mention.py` and `appendix/topic.R`. The paper's wording of
these results came from an earlier modelling regime (the review-phase controls of the analysis
records); the scripts here evaluate the same measures under the paper's final controls.

`results/` holds the outputs of a complete run (19 October 2026); `data/` is regenerated by the
measure scripts and is not kept in the repository.

**Environment.** R 4.5.3 with lme4 1.1-37, glmmTMB 1.1.15, ggplot2, patchwork; Python 3.12 with
torch 2.8, transformers 5.2, sentence-transformers 6.1, pandas, wordfreq; the repository modules
`scripts/entity_iv/{calculate_entity_iv,entity_roles,form_inventory,build_gum_transition_cost}.py`.

## Verification against the paper (19 October 2026)

Checked against the files the submitted PDF was built from:

* every measure script reproduces the original per-target values (exactly for count- and
  bit-based columns, to 1e-6 for the embedding-based ones, which were first computed on a GPU);
  the salience model reproduces its weights and AUC;
* the reading-time table reproduces the 40,387 observations with identical sums;
* the estimates reproduce Figure 1, Figure 2, Figure 3 and Table 1 on both datasets (Natural
  Stories to within 0.3 ms, CLASP to 1e-5); the held-out evaluation reproduces Tables 2 and 3 to
  the last printed digit;
* the appendix scripts reproduce the "Position" paragraph (d_est -78, d_beta +28, new entities
  +68 under the nonlinear controls; r = 0.97 between context mentions and position; -317 ms per
  SD), the "Linking cost" paragraph (+98 alone, +135 given the new entities, +90 given d_cos,
  d_beta +88 alone and +85 for the cost given it), the lead/lag pattern, the "Topic" paragraph
  (AUC 0.88; learned topic = most-mentioned entity in 85% / 93%; subject bit -0.033 rating points,
  14 AIC, +0.25 held out; 67 CLASP versions where the two bits part, subject bit significant
  under 7 strategies there, first-mention bit under none), and the CLASP surprisal-size and
  language-control paragraphs. The word-level localisation gives 4.9 / -1.3 / 3.3 / 4.0 / 5.0 ms
  per SD over the five regions against the paper's 5.1 / -0.8 / 3.5 / 4.0 / 4.7;
* the "Claims tested and not reported above" paragraph: the next-mention accounts on the 206
  targets whose first referent comes from the previous utterance rank grammatical (+69 ms per SD,
  t = 3.0), frequency (+44), information status (+38), recency (+25), and none beats the uniform
  expectation held out (paired gains within +-0.04 x 10^-2 nats, at most 6 of 10 stories); the
  Bayesian, Expectancy and Mirror pronoun models on the 104 targets that begin with a personal
  pronoun differ by 0.02 to 0.05 x 10^-2 nats, and the uniform count of compatible candidates is
  the only one significant under all 11 strategies; the referring-form distances have no effect on
  either dataset (|t| <= 1.8, held-out gains <= 0). The next-mention topic is the most-mentioned
  entity in 72% / 87% of the targets (paper: 73% / 87%) and its subject bit is -0.026 rating points
  per SD on CLASP, null on Natural Stories. These were re-evaluated under the paper's final
  controls; the paper's wording came from the review-phase regime of the analysis records.

Three places where the paper's text should be checked:

1. The "-84 ms per SD" of the number of established entities mentioned was fitted before the
   mentions-in-the-context control was added; with the paper's final controls it is -88 (here and
   in the analysis records).
2. The "+57" of the linking cost under the nonlinear position controls, the word-level numbers and
   the "r = 0.02" with d_cos come from an earlier implementation of the linking cost whose anchors
   were every non-pronominal context mention rather than one head lemma per entity (r = 0.99 with
   the beta of Eq. 10, which this code uses throughout); the nonlinear-position estimate is +62 here.
3. The sentence that the history version of d_part, and d_part itself, "add nothing" once the count
   of established entities mentioned and d_est are in the model has no computation behind it.
   `appendix/established_count.R` fits that model: d_part -54 ms per SD (t = -2.8, +5.7 AIC,
   significant under all 11 strategies), history version -48 (t = -2.6, +4.7 AIC). What the records
   do show is that the tightness and cohesion variants add at most 3 AIC given the count and d_est.
