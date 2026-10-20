# Entity information value: analysis record

Results of the entity information-value (IV) analysis on Natural Stories (reading times) and
CLASP (acceptability ratings), the specification behind every number, the changes made to the
pipeline and the R code to obtain them, and the files that hold the evidence. Numbers are from
the audit directory `../dual_mode_coherence_audit` (scripts `checks/*.R`, logs `out/*.log`,
figures `out/img/`); the repo code now reproduces the same quantities (see "Code changes").
Last updated 2026-10-04.

## 1. Specification (as decided)

| | Natural Stories | CLASP |
|---|---|---|
| response | sum of word reading times per sentence, ms (words 200 < RT < 2000, comprehension >= 4) | individual rating per participant and language variant, (rating - 1) / 3, ordered beta |
| rows | 40,387 sentence x reader rows, 485 sentences, 10 stories | 9,931 ratings, 500 sentences (100 contexts x 5 variants) |
| controls | summed surprisal + sentence length (characters), both linear | mean surprisal per word + input length (prefix + sentence words) + sentence position |
| surprisal LM | Pythia-70m-deduped step1000 | Pythia-12b-deduped final checkpoint (`scripts/score_clasp.py`, empty prefix = BOS only) |
| random effects | (1 \| WorkerId) + (1 \| story) | (1 \| WorkerId) (needed: -3,400 AIC without it; SD 0.39 logit) |
| IV channels | 14 released channels minus Euclidean + form-bigram transition + entity graph = 16 | same 16 |
| alternatives | 90 per item, 11 decoding strategies; temp_125 in the figures | same |
| estimates | one fit on all data; not averaged over folds | same |
| cross-validation | 5 folds by reader and leave-one-story-out; held-out Gaussian log-likelihood with predictive variance sigma^2 + tau^2 of the unseen group | 5 folds by participant and 10 folds by context; held-out ordered-beta log-likelihood |
| not in the model (decided) | sentence position; log / quadratic length; sentence random intercept | summed surprisal (fits 40 AIC worse than the per-word rate) |

Why the controls differ: for a judgement about a sentence the per-word surprisal rate is the better
predictor (checks/53: summed surprisal + sentence length is 40 AIC and 12 to 18 held-out nats worse,
and its two terms cancel each other); reading time accumulates over words, so the sum is right there.

## 2. Main results at temp_125 (figure `out/img/fig1_estimates_temp125`)

Estimate per SD of the predictor, with the IV block (controls + 16 channels) fitted on all data.
Natural Stories in ms (t), CLASP on the logit scale (z). "Alone" = the channel added on its own to the
controls, mean over the 11 strategies, with the number of strategies at p < .05.

| channel | NS, in block | NS, alone (sig/11) | CLASP, in block | CLASP, alone (sig/11) |
|---|---|---|---|---|
| surprisal | +831 (27) | | -0.105 (-11) | |
| length | +2,911 (91) | | -0.143 (-2.3) | |
| position | | | +0.135 (2.2) | |
| entity graph | -424 (-20.5) | -221 (11) | +0.038 (3.1) | +0.005 (0) |
| recurrence, exponential | +316 (12.8) | -122 (11) | -0.043 (-1.1) | -0.011 (1) |
| recurrence, uniform | +18 (0.5) | -159 (11) | +0.110 (2.2) | -0.012 (1) |
| new head | +181 (12.7) | +289 (11) | -0.015 (-1.8) | -0.001 (0) |
| transition, role (released) | -5 (-0.2) | -140 (11) | -0.104 (-3.1) | -0.020 (11) |
| transition, form (bigram) | -186 (-7.4) | -163 (11) | -0.052 (-1.5) | -0.017 (11) |
| lexical 1-gram | -98 (-5.3) | -28 (8) | -0.028 (-2.3) | +0.020 (9) |
| lexical 2-gram | +100 (3.9) | +84 (11) | +0.060 (2.7) | +0.025 (11) |
| lexical 3-gram | +21 (0.9) | +81 (11) | -0.010 (-0.5) | +0.020 (10) |
| POS 1-gram | +32 (1.8) | -50 (11) | +0.017 (0.9) | +0.014 (1) |
| POS 2-gram | +31 (1.3) | -107 (11) | -0.020 (-0.8) | +0.007 (0) |
| POS 3-gram | -32 (-1.4) | -85 (11) | +0.005 (0.2) | +0.013 (4) |
| deprel 1-gram | -98 (-5.2) | -89 (11) | -0.033 (-1.9) | +0.011 (0) |
| deprel 2-gram | -123 (-5.4) | -94 (11) | +0.047 (2.0) | +0.014 (3) |
| deprel 3-gram | +66 (3.0) | -39 (11) | -0.021 (-1.0) | +0.013 (4) |
| cosine | +93 (10.4) | +49 (11) | -0.030 (-2.3) | +0.014 (3) |

Reading the two columns per dataset: in Natural Stories every channel alone is significant in all
11 strategies because every channel tracks how far the story has progressed (section 4); the block
estimates are what remains given the other channels. In CLASP the entity channels are near-copies of
each other at the item level (recurrence uniform/exponential r = 0.97, role/form transition 0.97,
recurrence vs transition 0.95, graph vs recurrence 0.8; 20 of 100 contexts have no established
entity, so all entity IVs are 0 together), so their block coefficients are a suppression pattern;
the "alone" column is the one to interpret there: the two transition channels (about -0.02 logit per
SD, a sixth of surprisal's effect) and the positive lexical 2/3-gram. Graph, recurrence and new head
have no effect of their own in CLASP. The CLASP transition effect is partly the empty-context contrast:
with a dummy for those 20 contexts it is -0.067 (7/11), within the 80 real contexts -0.040 (7/11).

Across strategies (`out/img/fig7_table_*`, `out/table_across_strategies.txt`): no channel that is
significant in all 11 strategies flips sign; temp_125 and typical_02 are the extremes. The n-gram
channels are weakest at temp_125 because the flattened token distribution pushes their overlap
distances to the ceiling (lexical 2-gram mean 0.98, SD 0.02 vs 0.96 / 0.04 elsewhere); entity
channels are unaffected.

## 3. Mention counts versus entity IV (figures `fig4_counts_overlap`, `fig5_entity_with_counts`)

AIC gain over the control model at temp_125; "counts" = the seven gold covariates (mentions,
transitions and entities in the target; context mentions, context entities, context role cost,
cumulative role cost) from `count_entities_*.py`.

| | Natural Stories | CLASP |
|---|---|---|
| IV block (16 channels) | 1,606 | 34 |
| gold counts | 2,614 | 14 |
| both | 3,117 | 31 |
| unique to IV / shared / unique to counts | 504 / 1,103 / 1,511 | 17 / 17 / -3 |
| entity sub-block alone -> given counts | 1,113 -> 108 (90 % shared) | 27 -> 20 (27 % shared) |
| graph alone -> given counts (all strategies) | 479 -> 45 AIC; -224 -> -74 ms/SD | |

Entity channels with and without the counts (temp_125):

| channel | NS, IV only | NS, with counts | CLASP, IV only | CLASP, with counts |
|---|---|---|---|---|
| entity graph | -424 (-20.5) | -151 (-6.7) | +0.038 (3.1) | +0.033 (2.3) |
| recurrence, exponential | +316 (12.8) | +26 (1.0) | -0.043 (-1.1) | -0.033 (-0.8) |
| new head | +181 (12.7) | +68 (4.7) | -0.015 (-1.8) | -0.010 (-1.1) |
| transition, form | -186 (-7.4) | -98 (-3.9) | -0.052 (-1.5) | -0.057 (-1.6) |
| transition, role | -5 (-0.2) | -40 (-1.4) | -0.104 (-3.1) | -0.109 (-2.9) |
| recurrence, uniform | +18 (0.5) | +139 (4.0) | +0.110 (2.2) | +0.107 (2.0) |

The graph distance (checks/57): higher when the target re-mentions FEW context entities in a
discourse with MANY established ones (r = -0.30 with the target's re-mentions, +0.36 with the number
of established entities); its faster-reading effect is not reducible to those counts (-141 ms/SD,
t -13, with established entities, re-mentions and new entities in the model). Its sign is the
opposite of a processing-cost reading, and in CLASP the same channel has no effect of its own.

### The entity-graph distance, formally

Not a graph edit distance: a cosine distance between connection profiles on the entity graph of
Guinaudeau & Strube (2013). With context sentences s = 1..n and established entities E:

1. Context matrix B (n x |E|): B[s, e] = max over mentions of e in sentence s of w(role), with
   w(subject) = 3, w(object) = 2, w(other) = 1; 0 if e is not mentioned in s.
2. Continuation vector b_c (|E|): b_c[e] = max over mentions of e in the continuation c of w(role);
   0 if c does not mention e. New entities in c do not enter.
3. Connection profile g_c = B b_c, i.e. g_c[s] = sum_e B[s, e] b_c[e]: how strongly c links back to
   context sentence s through shared entities, role-weighted on both ends.
4. d(t, a) = 1 - (g_t . g_a) / (|g_t| |g_a|) for the target t and an alternative a; d = 0 if both
   profiles are zero, d = 1 if exactly one is. Aggregated over the 90 alternatives as mean / p80 / min.

The cosine ignores scale, so d measures WHERE a continuation attaches to the context, not how
strongly: 0 when target and alternative spread their links over the same earlier sentences in the
same proportions, 1 when the target links to none of the established entities while the
alternative does. The minimum over alternatives is ~0 (some sample always links like the target);
the mean carries the signal and grows with how spread out the samples' links are, hence with the
size of the established cast. `d8_graph_recent` (computed in the audit, not reported) weights row s
of B by 1 / (n - s + 1).

## 4. Held-out log-likelihood (figure `fig3_heldout`; nats per row, mean over strategies)

| | NS new readers (5 folds) | NS new stories (LOSO) | CLASP new participants | CLASP new contexts |
|---|---|---|---|---|
| IV block | +0.0042 | -0.011 | -0.0007 | 0.000 |
| gold counts | +0.0066 | +0.058 | -0.0006 | +0.0013 |
| both | +0.0082 | +0.052 | -0.0020 | -0.0005 |

Per channel alone in Natural Stories (checks/37, predictive variance for the unseen group): graph
+0.017 nats/row on new stories, exponential / uniform recurrence +0.008 / +0.007, form transition
+0.004, new head -0.026 (negative in all 11 strategies; it is the reason the block loses on new
stories; the block without new head gains +72 nats, with graph +499). On new readers every entity
channel is positive (graph +0.002). With the Rmd's original sigma-only likelihood all blocks look
worse than the baseline on new readers; that was the variance mis-specification, now fixed.
CLASP: nothing transfers (all blocks within +-0.003 nats per rating).

Comparison: Meister, Giulianelli & Pimentel (2024) also use held-out Gaussian log-likelihood, but with
plain linear regression, random word-level folds (10) and per-word units (10^-2 nats); our by-reader
block gain is of the same order as their headline gains. With 10 grouped folds the smallest paired
permutation p is 0.002; with 5 it is 0.06. No paired fold test is reported here.

## 5. Transition cost models: bigram vs MLP (figure `fig6_transition_variants`; checks/54)

Each variant swapped into the full IV block for its bigram counterpart (mean over strategies):

| | NS estimate (sig/11) | NS dAIC vs bigram | CLASP estimate (sig/11) | CLASP dAIC vs bigram |
|---|---|---|---|---|
| role, bigram (released, deprel states) | +6 (1) | ref | -0.097 (11) | ref |
| role, MLP | -35 (3) | +2.3 (better in 8) | -0.079 (10) | -1.7 (worse in 11) |
| form, bigram | -133 (11) | ref | -0.045 (1) | ref |
| form, MLP | -103 (11) | -9.6 (worse in 11) | -0.068 (9) | +3.7 (better in 11) |

All four together in one model produce suppression pairs (NS role bigram +176 / role MLP -184), so
the swap is the only readable comparison. GUM held-out cost per transition (bits): role bigram vs
MLP and form bigram vs MLP are stored under `cv` in `scripts/entity_iv/gum_transition_models.json`.

## 6. Surprisal model

Natural Stories IV estimates are practically identical across Pythia-410m / 1.4b / 6.9b / 12b at
step1000 and final (eight files scored with `scripts/score_stories.py`, Rmd run end to end on all):
recurrence uniform -264 to -338, new head +184 to +209, deprel 2-gram -182 to -255, cosine +59 to
+89; only the surprisal coefficient moves (+617 step1000 to +1,319 for 12b final). The reported
Natural Stories numbers use Pythia-70m step1000 by decision. CLASP uses Pythia-12b final; the
Pythia-12b and Gemma-2-9B per-word surprisals correlate 0.72 over the 500 sentences and give the same
channel picture.

### Surprisal model, control fit (checks/64)
GPT-2 small (1,000-token sliding window) fits the Natural Stories control model BETTER than Pythia-70m step1000 (AIC
715,520 vs 715,840; +829 held-out nats on unseen stories, +67 on unseen participants); Pythia-12b final better still
(715,280). Pythia-70m step1000 is kept by decision (Oh & Schuler 2023); the channel estimates do not depend on it.

### Most prominent mention instead of first mention (checks/63; ext/iv_prominent.py)
The graph channel represents a multiply mentioned entity by its most prominent mention; the released new-entity and
transition channels by its first. Recomputed with the most prominent mention and swapped into the IV block: NS transitions
+24.9 AIC (better under all 11 strategies; role transition -41 ms/SD, 6/11, vs +6, 1/11; form -150 vs -133), new entities
-6.8 AIC (r = 1.00 with the first-mention version); CLASP transitions +0.8 (8/11), new entities unchanged. A max-over-mentions
cost is +11.8 in NS and -6.7 in CLASP. Reported results keep the first-mention definition.

## 7. Anticipatory entity measures (Pimentel et al. 2023 framing; checks/46 to 52)

Natural Stories only (CLASP: context-level measures, no effect of the entity count on ratings,
t -0.3).

| measure | alone (ms/SD, t) | given Hartley + log position + 7 counts | held-out on top of those controls |
|---|---|---|---|
| Hartley entropy = log2(1 + established entities) | -395 (-45), 1,948 AIC | (is a control) | +1,083 nats LOSO alone |
| next-subject distribution, Renyi-1/2 (LM samples) | -408 (-46), 2,090 AIC | t -1.6 | |
| per-entity re-mention uncertainty, GUM logistic model | +338 (39) | +106 (6.7), +43 AIC | +174 nats LOSO, +14 by reader |
| per-entity re-mention uncertainty, from the samples | +345 (39) | +86 (5.6), +29 AIC | +113 / +14 |
| next-referent softmax entropy, GUM-tuned encoder (ModernBERT scorer) | -389 (-37) | -105 (-7.0), +47 AIC | -118 nats LOSO (does not transfer) |

Hartley entropy is the number of established entities: it predicts FASTER reading, it is
inseparable from story position (log position alone 2,106 AIC; Hartley adds nothing on top of it),
and any entropy over a growing candidate set inherits it. The one measure with the anticipation-cost
sign that survives every control and transfers to new stories is the mean per-entity re-mention
uncertainty (binary entropy of P(entity returns in the next sentence), averaged over the established
entities; anticipation beats realisation, t 4.4 vs 1.5). The encoder softmax (prefix-only
ModernBERT-base entity vectors, MLP scorer, one "new entity or no mention" outcome; GUM 5-fold NLL
2.96 bits vs 4.07 for the base-rate baseline) is the only distribution entropy that survives the
controls, but its sign is facilitation (expected established entity -> faster) and it is worse than the
controls on held-out stories. CorPipe's own antecedent distributions are per mention, not
anticipatory; that line was dropped.

## 8. Bugs found in the pushed R code and what was done

| where | bug | fix in the repo |
|---|---|---|
| naturalstories.Rmd | IV `sentence_index` (0-based) joined to surprisal `sentence_id` (1-based): every rating matched to the NEXT sentence's IV | `sentence_id = sentence_index + 1` + `stopifnot` no NA |
| naturalstories.Rmd | held-out likelihood used `test_data$Reaction` (does not exist): RSS = 0, test data never entered; `+(n/2) log 2 pi` sign wrong | observed response, correct sign, and predictive variance sigma^2 + tau^2 for the held-out group |
| both Rmds | cosine and Euclidean both in the model; Euclidean = sqrt(2 cosine) per pair (VIF > 200) | Euclidean dropped |
| both Rmds | estimates / SE / p averaged over CV folds | one full-data fit |
| naturalstories.Rmd | `sentence_length = length(zone)` = words x readers | `n_distinct(zone)` |
| both Rmds | no `set.seed` before fold assignment; `rm.na` typo | seeded; `na.rm` |
| clasp.Rmd | ratings and IV averaged over the 5 language variants of a context | one row per rating, joined on context + language |
| clasp.Rmd | CV loop ran the mention formula only; 76 rows sampled instead of 76 workers; y / mu indexed differently in the ordered-beta likelihood | loop runs iv / mention / both; same rows for y and mu |
| clasp.Rmd | `ctx_n_mentions_s` inside the IV formula | moved to mention / both formulas |
| score_clasp.py | empty pre-context read as NaN and scored as the string "nan" for 2,000 ratings | fixed in the scorer |
| naturalstories.Rmd (data) | 20 CLASP contexts with an empty pre-context (all entity IVs 0, surprisal without context, rated as high as the rest) | documented; dummy / 80-context checks in checks/55 |
| information-value repo | `to_jsonl.py` swaps the in-context and out-of-context CLASP ratings | to be raised with the authors |

Left as design decisions: word-level RT filter before summing; reader + story random effects only;
folds by reader (plus leave-one-story-out); linear length; no position in Natural Stories.

## 9. Code changes in this repo (working tree, uncommitted)

- `scripts/entity_iv/entity_iv_ext.py` (new): `d3_transition_form`, `d3_transition_role_mlp`,
  `d3_transition_form_mlp`, `d8_graph`; `calculate_entity_iv.py` appends them after the original
  45 columns (`--transition-models`, default `gum_transition_models.json`; `--no-ext-channels` to
  skip). The original columns are computed by unchanged code.
- `scripts/entity_iv/form_inventory.py`, `build_gum_models.py`, `gum_transition_models.json` (new):
  the 15-state form inventory and the GUM-fitted bigram / MLP cost models.
- `scripts/score_clasp.py`: empty-prefix fix (an empty pre-context is scored with the BOS token only,
  not with the string "nan"); the CLASP Pythia-12b files used here were scored with this fix.
- `analysis/naturalstories.Rmd`, `analysis/clasp.Rmd`: section 8 fixes; new channels in the IV
  formula; gold counts merged (NS) and `mention_formula` / `both_formula` in both; CV loop over the
  three formulas, tagged `formula`, with `cv_group` for the two fold schemes; predictive-variance
  held-out likelihood (NS); 10 folds by context added (CLASP). Output file names unchanged
  (`viz.Rmd` still expects the old names; to be fixed separately).
- Verification (checks/60_identity_check.py, `out/verify_chain.log`): the modified
  `calculate_entity_iv.py`, run on Natural Stories (485 items) and CLASP (500 items) at temp_125,
  reproduces every released non-semantic column byte for byte (39 of 45 columns, 0 differing cells)
  and the audit's values for all 12 new columns exactly. The six `d_semantic_*` columns differ from
  the released file in the 7th decimal (max 9e-7 absolute) because the embedding model ran on this
  node's CPU rather than the original GPU: the unmodified HEAD script, run in the same environment on
  story 1, gives cells identical to the modified script and the same 312 semantic-cell differences
  against the released file. The semantic code was not changed. Repeating the run on an H200 (audit venv,
  torch 2.8; and the original `corpipe26` environment, now torch 2.10 / transformers 5.2) still leaves
  83 to 478 differing semantic cells per column, max 3.6e-7: float32 embedding arithmetic is not
  bit-stable across torch builds and GPU kernels, and the exact versions of the original run are not
  recorded. Irrelevant for the estimates (predictors are standardised).
- End-to-end check of the modified Rmds (scratch copy, `../dual_mode_coherence_audit/scratch_run/analysis/`,
  reading the new IV files): `naturalstories.Rmd` runs through with the 70m surprisal file and gives the
  figure-1 estimates to the unit (graph -424, form transition -186, exponential recurrence +316, new head
  +181, lexical 2-gram +100, cosine +93); its CV table now has three formulas (iv / mention / both) x two
  fold schemes. `clasp.Rmd` (Pythia-12b file only) runs through as well and gives the CLASP figure
  estimates (role transition -0.104, uniform recurrence +0.110, lexical 2-gram +0.060, graph +0.040);
  its CV table has the three formulas x (5 folds by participant, 10 by context). The scratch runs
  skip DHARMa plots and stand in for two packages not installable on this node (`this.path`,
  `ordbetareg`, whose `dordbeta` the held-out likelihood calls); the Rmds themselves are unchanged in
  those respects.

### Model selection by held-out likelihood (checks/69) and entity-only counts comparison (checks/67)
Forward selection (add the channel that most improves held-out LL; NS unseen stories, CLASP unseen contexts; temp_125): NS selects
graph (+2.3 x10^-2 nats/obs), lexical 3-gram, deprel 2-gram, cosine, lexical 1-gram (five channels; no recurrence / new-entity /
transition channel). Nested (selection inside each left-out story): selected model +1.6 (range -6.8..+10.6), graph chosen in 10/10
folds, cosine 9/10; counts alone +6.4; selected + counts +6.3. CLASP: no channel improves -> control model.
Counts comparison redefined as controls + 6 entity channels (+/- 7 counts): NS entity block 1,183 AIC, 85% shared with counts;
temp_125 estimates E -> EC: recency recurrence +325 -> +36, new entities +256 -> +128, form -178 -> -106, graph -445 -> -191,
uniform recurrence +44 -> +190. CLASP: entity block 12 AIC, counts 11, two thirds shared; role transition -0.064 given counts
(5/11 sig). Rmd both_formula = controls + entity channels + counts. Surprisal family (checks/70): the IV gain does not shrink as
the surprisal model improves (16 channels 1,513-1,721 AIC across Pythia 70m..12b and GPT-2 small; graph alone 581 -> 760);
final checkpoints fit sentence RTs better than step1000; whole-word LSTM (Gulordava 2018) fits far worse (-826 AIC).
Tokenisation bias: score_stories.py has no leading-whitespace correction (score_clasp.py does); applying it (ext/score_stories_bow.py)
leaves sentence sums identical (r = 1.000, mean diff -0.006 nats), control AIC within 6-14, channel estimates unchanged; at the WORD
level it is slightly worse (checks/71: 70m step1000 dAIC 4,596 vs 4,777).
Word level vs sentence level (checks/71 vs 70): at the word level (RT per word, surprisal + spillover over length/frequency/position
baseline) inverse scaling holds: 70m step1000 best (4,777 AIC), all step1000 > all final, final falls 3,876 (70m) -> 2,370 (12b),
GPT-2 small 3,943; whole-word LSTM 1,081, word Transformer-XL (WT103, 257M) 1,713. At the sentence level the order reverses (final
> GPT-2 > step1000 > whole-word; 12b final best, +562 AIC). No larger whole-word LM is publicly available (lm_1b checkpoints 403).
The sentence-level reversal is a property of the sum + length control pair (checks/73): the sum is nearly length (r 0.95 for 70m,
0.89 for 12b), and the better model's sum duplicates length least; with mean surprisal + length, or the sum alone, the word-level
order returns (mean + length: 392 AIC for 70m step1000 vs 140 for 12b final). The channel gain is unaffected by the choice.
Gemma-2-9b / 27b (bf16): control +312 / +310 AIC vs 70m step1000 (less than Pythia-12b's +562); the 16 channels keep 1,523 / 1,543
of their 1,606 AIC; entity estimates unchanged. So no surprisal model, up to 27B parameters, absorbs the entity channels.

## 10. Paper (`../dual_mode_coherence_audit/paper/`)

Revision of 2026-10-06: everything from Experimental Setup on rewritten (see FINAL_RESULTS.md); compiled with Tectonic
(`bin/tectonic`, ACL style files in `paper/build`); `main.pdf` saved next to `main.tex`. Main held-out table collapses the ten
non-entity channels into one step; the full per-channel table (by participant) sits on the appendix float page with the two
strategy grids. CLASP unseen-participant held-out now marginalises over the participant distribution (checks/61 SCHEME=
unseen_participants, checks/67); earlier conditional numbers (R2 -0.11) superseded.

`main.tex` (assembled from `main_template.tex` by checks/66_assemble_tex.py, tables from checks/65_latex_tables.py and
the nested held-out run checks/61), `fig1_estimates_temp125.pdf` (16 channels, controls in the caption, paper channel
names), `fig5_entity_with_counts.pdf`, `fig1_controls.csv`. Changes against the draft: the two held-out tables filled
(predictors added in turn; dLL in 10^-2 nats per observation, held-out R2, RMSE; NS unseen stories / participants, CLASP
unseen contexts / participants), the held-out paragraph rewritten from them, the claim that Pythia-70m step1000 beats
GPT-2 small removed and corrected in the appendix (GPT-2 small fits the control model better by 320 AIC), a footnote and
appendix paragraph on first vs most prominent mention, Appendix B on anticipation without alternatives (Hartley entropy
= count; per-entity re-mention uncertainty), captions rewritten. Needs a `pimentel-etal-2023-effect` entry in custom.bib.

## 11. Where things are

Audit directory `../dual_mode_coherence_audit`: `FINAL_RESULTS.md` (running record),
`BUGS_AND_FIXES.txt`, `SLACK_CLASP_CHANGES.txt` (plain-language summary for the collaborator),
`checks/34..60`, `out/*.log`, `out/img/fig1..fig7`, `out/iv_ext/`, `out/iv_new/` (new channel
files for all 11 strategies), `scratch_run/` (end-to-end Rmd runs and the locally scored Pythia
surprisal files: eight Natural Stories files, Pythia-12b CLASP per-worker files).
