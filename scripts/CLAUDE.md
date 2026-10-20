# CLAUDE.md

## Task

Verify the results of the entity information-value (IV) pipeline
(`scripts/entity_iv/`) and the k-fold regression analysis (`analysis/`).
Output: a findings report plus the local scripts that produced the evidence.

**This is read-only with respect to the repo.** Nothing in the repo is changed,
added, deleted, reformatted, or committed. All work happens in a scratch
directory outside the repo (see "Isolation").

Terms:
- **Target**: the sentence that actually occurred next.
- **Alternative**: a sentence sampled from Gemma-2-9B given the same context.
- **IV**: distance between target and each of N alternatives, aggregated
  (`_mean`, `_p80`, `_min`). Higher = target less like what the LM expected.
- **Item**: one target plus its context. Natural Stories (NS): (story, sentence).
  CLASP: (context id, language variant).
- **Channel**: one of the 15 distance functions in `METRICS`
  (`scripts/entity_iv/calculate_entity_iv.py`).

## Isolation (mandatory)

```bash
export REPO=$(git rev-parse --show-toplevel)
export AUDIT="$REPO/../dual_mode_coherence_audit"   # outside the repo
mkdir -p "$AUDIT"/{checks,out,copies}
export PYTHONDONTWRITEBYTECODE=1                    # no __pycache__ written into the repo
export UV_PROJECT_ENVIRONMENT="$AUDIT/.venv"        # venv outside the repo
git -C "$REPO" --no-optional-locks status --porcelain > "$AUDIT/git_status_start.txt"
git -C "$REPO" rev-parse HEAD >> "$AUDIT/git_status_start.txt"

uv sync --frozen --no-install-project --project "$REPO"   # --frozen: never rewrite uv.lock
uv run --frozen --project "$REPO" python "$AUDIT/checks/<name>.py"
Rscript "$AUDIT/checks/<name>.R"
```

Rules:
1. Allowed git commands: `status`, `log`, `diff`, `show`, `blame`, `ls-files`
   (use `--no-optional-locks`). Not allowed: `add`, `commit`, `checkout`,
   `switch`, `branch`, `stash`, `reset`, `restore`, `pull`, `push`, `clean`.
2. Do not use edit/write tools on any path under `$REPO`.
3. Python: import repo modules via `sys.path.insert(0, f"{REPO}/scripts/entity_iv")`.
   R: `source(file.path(REPO, "analysis/k_fold_cross_validation.R"))`.
4. To test a corrected version of repo code, copy the function or file into
   `$AUDIT/copies/` and edit the copy. Label copies with the finding ID.
5. Never run the Rmds in place: they `write.csv` into `analysis/results/`.
   Extract with `knitr::purl(input, output = file.path(AUDIT, "copies", "<name>.R"))`,
   redirect every `write.csv` / `ggsave` to `$AUDIT/out/`, and use absolute
   input paths.
6. Do not run pipeline stages 1 or 2 (GPU, hours) or rescore surprisal.
   Work from existing outputs. If a check needs a GPU, write it and mark it
   "not run".
7. End of session: rerun the status command into `git_status_end.txt`. It must
   match `git_status_start.txt`. Report any difference.

Other rules:
- Do not change a channel's definition to make a test pass. Record it and ask.
- Every finding needs evidence: a failing assertion, a count, or a printed
  example with file and function name.
- If required data is missing, stop and ask for the path. Synthetic fixtures only
  in unit tests, labelled synthetic.
- Plain language in the report. Define a term the first time it appears.

## Pipeline map

```
NS manual CSVs / CLASP processed_ratings.csv
 [1] generate_{ns,clasp}_vllm.py            -> raw_*_vllm/*.jsonl         (GPU)
 [2] run_parallel.py -> annotate_{ns,clasp}.py, annotate_common.py,
     corpipe26_seeded.py                     -> annotated_alternatives/*/*.json
 [3] calculate_entity_iv.py                  -> iv_values/<dataset>/iv_<strategy>.csv, iv_config.json
 [4] mention_role_costs.py, count_entities_{ns,clasp}.py  (gold counts; independent of [3])
     build_gum_transition_cost.py            -> gum_deprel_costs_50.json (used by [3] and [4])
 [S] scripts/score_stories.py, score_clasp.py -> annotated/surprisal/...
 [R] analysis/{naturalstories,clasp}.Rmd + k_fold_cross_validation.R
                                             -> analysis/results/*.csv, estimate figure
```

`scripts/iv_pipeline/` was an older, unused version of the pipeline and was removed on
2026-10-19; `paper/` holds the code behind the submitted paper. The committed Rmds use the old names and 6 IV
predictors. The estimate figure (17 predictors, `d_syntactic_dependency_*`,
`input_length_s`, `surprisal`) was made by code that is **not in the repo**.
Ask the user for it, and establish which code produced each file the analysis
reads before auditing it.

## Surprisal baselines used in the figure (stated by the user)

| Dataset | Surprisal LM | Checkpoint |
|---|---|---|
| CLASP | Pythia-12b (deduped) | confirm: `step1000` or final |
| NS | Pythia-70m (deduped) | confirm: `step1000` or final |

The baselines differ in strength, so "IV gain over surprisal" is not directly
comparable between the two datasets (see G9).

## Data not in git (ask the user; record answers in `$AUDIT/PATHS.md`)

- NS manual CSVs `story_*_entity_annotated.csv`
- `naturalstories/naturalstories_RTS/processed_RTs.tsv`, `naturalstories/parses/ud/stories-aligned.conllx`
- CLASP `processed_ratings.csv`, CLASP surprisal TSVs (Pythia-12b)
- `raw_*_vllm/`, `annotated_alternatives/`, `iv_temp_125_union/`
- `annotated/surprisal/naturalstories/*.tsv`
- The script that produced the estimate figure, and the k used there

## Invariants (one check each)

| ID | Invariant |
|---|---|
| I1 | NS `sent_id` / `sentence_index` is 0-based and contiguous in manual CSVs, IV files, and count files. |
| I2 | Surprisal TSV `sentence_id` is **1-based** and comes from the UD conllx segmentation (`_load_naturalstories_conllx`), not the manual CSVs. |
| I3 | All `d_*` are distances, 0 = identical. `d_semantic_cosine = 1 - cos`. `d_lexical_*`, `d_pos_overlap_*`, `d_syntactic_dependency_*` are `1 - overlap`. |
| I4 | Embeddings are L2-normalised, so per pair `d_semantic_euclidean == sqrt(2 * d_semantic_cosine)`. |
| I5 | Empty vs empty gives 0 in every Jaccard-type channel. |
| I6 | Exactly N alternatives per (item, strategy); duplicates kept. State the N used by the analysis (README says 90, script default 100). |
| I7 | Same `--role-inventory` in the cost fit, stage 3, and stage 4. |
| I8 | CLASP: 5 targets per context id (English, Czech, German, Spanish, French round-trip variants), each with its own rating. |

## Checklist

`[S]` = suspect found while reading the code; verify first. `[C]` = routine check.

### A. Generation (stage 1, read outputs only)

- A1 [C] Prompt for NS sentence k contains sentences 0..k-1 only; target not in
  prompt; k=0 uses `" "`. Same for CLASP `Pre-Context`.
- A2 [S] NS prompt is built from space-joined CSV tokens (`"England , you"`), not
  natural text (`generate_ns_vllm.read_sentences`). Check whether alternatives
  inherit that style and whether target text in stage 3 uses the same rendering.
  Quantify the effect on `d_lexical_*`.
- A3 [C] Strategy names match sampling params in `STRATEGIES`.
- A4 [C] Seeds are deterministic per (model, item, strategy).

### B. Filtering and annotation (stage 2, read outputs only)

- B1 [S] `--incomplete-fallback` is on by default. From `input_qa` in the part
  files, report per item/strategy the share of `incomplete`, `held_back_*`, and
  `padding_duplicate` alternatives. Correlate with IV.
- B2 [S] Annotation symmetry. Read the stored `settings` (`target_annotation`,
  `context_annotation`, `ns_syntax`). If the NS target uses gold syntax or gold
  mentions while alternatives use Stanza + CorPipe, `d_syntactic_dependency_*`
  and d1-d3 compare two different annotators. `iv_config.json` hard-codes
  `"syntax_source": "supplied gold Natural Stories CSV ..."`; check it against
  the actual settings.
- B3 [C] Exact-copy test: alternatives whose normalised text equals the target
  should have 0 on all 15 channels (semantic within 1e-6).
- B4 [C] Every context cluster id used by a target/alternative exists in the
  context. New clusters do not reuse context ids.
- B5 [C] Coverage: one item per NS manual-CSV sentence and per CLASP
  (id, language). No duplicates after the `_union` merge.

### C. IV computation (stage 3)

Recompute from `annotated_alternatives/` into `$AUDIT/out/` only on CPU and only
for a sample of items unless the user says otherwise.

- C1 [C] Per channel: identity (0), symmetry, bounds ([0,1] Jaccard-type,
  [0,2] cosine and Euclidean), empty/empty, John/Mary example in `DISTANCE.md`.
- C2 [C] Verify I4 on a sample of pairs; report Pearson r between
  `d_semantic_cosine_mean` and `d_semantic_euclidean_mean`.
- C3 [C] `quantile()` equals `numpy.quantile(..., method="linear")`.
- C4 [C] `recurrence_distances` exponential weights vs brute force.
- C5 [S] `transition_distance` scores an absent entity as 0 bits (the docstring
  of `transition_distance_absent` calls this wrong). Count affected comparisons.
- C6 [C] Recompute `gum_deprel_costs_50.json` cells from `counts`/`row_totals`.
- C7 [C] Correlate each `*_mean` with target length, `n_target_entities`,
  `n_context_entities`, sentence position, and B1 fallback share.
- C8 [C] Permuted baseline: IV against alternatives from a different item should
  on average exceed true IV for entity channels.
- C9 [C] Released IV CSVs match a local recomputation for the sampled items.

### D. Surprisal

- D1 [S] Committed `annotated/surprisal/pythia-70m-deduped-step1000-full.tsv` has a
  6-name header but 5-field rows; `read.csv` would put surprisal under
  `sentence_position`. Confirm the file actually used is the 5-column format.
- D2 [C] `(item, zone)` is unique in the NS surprisal file; the merge with RTs
  does not duplicate rows.
- D3 [C] Subword-to-word aggregation (`_get_word_scores`), BOS handling, units.
- D4 [C] Confirm the table above. CLASP `input_length` is counted in words
  (`str_count(" ") + 1`) but the figure label says chars.

### E. Joins (highest risk)

- E1 [S] NS off-by-one. `naturalstories.Rmd` renames IV `sentence_index`
  (0-based) to `sentence_id` and merges with the surprisal-derived `sentence_id`
  (1-based) on `(item, sentence_id)` with no offset. Test: print RT-side
  sentence text next to IV target text for several keys. Also compare the two
  segmentations (conllx vs manual CSVs): sentence count and token sequence per
  story.
- E2 [S] CLASP language mismatch. `clasp.Rmd` averages IV over the 5 language
  variants per `document_id` before joining on `sent_id`, while ratings and
  surprisal are per variant (`lang_id`). Map `lang_id` to IV `language`; report
  how much IV varies across variants of one context.
- E3 [C] After every merge: rows before/after, unmatched keys, NA count per
  predictor (models drop NA rows silently).
- E4 [C] Id parsing: `str_sub(document_id, 7, 10)`, `str_split_i(document_id, "_", 2)`.

### F. Regression and cross-validation code

- F1 [S] `calc_test_loglik_lmer` uses `test_data$Reaction` (left over from the
  `sleepstudy` example). The column does not exist, so the residual sum of squares
  is 0 and `ll/n = 0.5*log(2*pi) - log(sigma_train)`; the `log(2*pi)` sign is
  also wrong. Reproduce `ll0` in `naturalstories_cv_summary.csv` from the training
  sigma to confirm. Corrected local copy: `sum(dnorm(y, mu, sigma(fit), log = TRUE))`.
- F2 [S] `calc_test_loglik` divides by `length(test_data)` (number of columns).
- F3 [S] `r2()` in the CV loop is computed on training data.
- F4 [S] Folds grouped by `WorkerId` test new readers; IV predictors are
  sentence-level. Also group by sentence (NS `item:sentence_id`, CLASP `sent_id`).
- F5 [S] No by-sentence random intercept (CLASP `(1|WorkerId)` only; NS `item`
  is the story). Sentence-level values repeat across workers, so SEs are too small.
- F6 [S] `fixef_summary` averages estimate, SE, CI and p over folds; not a valid
  test. `tidy()` on `lmer` gives no p-values: find the source of the NS stars.
- F7 [S] Model selection on `expl_df`, then CV on data including it. CLASP
  `sample(pull(WorkerId), 75)` samples rows with repeats. NS CV has no `set.seed`.
- F8 [C] Scaling by column ranges (`a:b`) depends on column order; scaling is
  done before the fold split.
- F9 [S] NS `sentence_length = length(zone)` counts words × workers.
- F10 [S] NS word-level outlier removal happens before summing per sentence.
- F11 [S] `mean(x, rm.na = TRUE)` (should be `na.rm`).
- F12 [S] Collinearity: run `check_collinearity()` on the figure model; report VIF.
- F13 [S] Figure labels: "Cosine similarity" is a distance; "... overlap" are
  `1 - overlap`; "Eculidean" typo. Check whether `surprisal` is unscaled.
- F14 [C] NS: `sum_RT` in raw ms with a Gaussian model; DHARMa residuals.

### G. CLASP: are the null results a sample-size or CV artifact?

User's question: most CLASP IV effects are small and not significant after
k-fold CV; is this because CLASP is small and each fold has only a few hundred
items? Answer with the checks below. Work on local copies with F1/F2 corrected.

- G1 [C] Confirm from `model_cross_val` that coefficients and p-values come from
  the training fit (k-1 folds), not the test fold. Test-fold size affects only
  held-out Δll and RMSE. Note: committed `clasp.Rmd` uses `num_folds = 5`.
- G2 [C] Fit the figure model once on all data, with `(1|WorkerId)` and with
  `(1|WorkerId) + (1|sent_id)`. Report estimate, SE, 95% CI per term next to the
  fold-averaged values.
- G3 [C] Per term: SD of the estimate across folds and share of folds whose sign
  matches the full-data fit.
- G4 [C] Held-out gain: pool test log-likelihood over all folds, Δll per rating,
  95% CI by bootstrap over `sent_id` (and separately over `WorkerId`). Repeat CV
  with 5 seeds; report the spread. Do it for folds grouped by `WorkerId` and by
  `sent_id`.
- G5 [C] Report n rows, n distinct `sent_id`, n workers. Per term, minimal
  detectable effect `MDE ≈ 2.8 × SE` (two-sided α = .05, power .8) from G2.
- G6 [C] Estimates are on the logit scale (beta family). Report the average
  marginal effect on the 0–1 rating scale, `mean(β · μ̂(1 − μ̂))`, per term,
  with surprisal as reference.
- G7 [C] Cosine + Euclidean: report `β_cos + β_euc` with its SE from `vcov`, and
  refit with each one alone.
- G8 [C] Refit with one channel per family (entity, lexical, POS, deprel,
  semantic); compare AIC and held-out Δll with the 15-channel model.
- G9 [C] Only with surprisal files that already exist locally: rerun G2/G4 with a
  different surprisal model per dataset and report whether IV estimates change.
- G10 [C] Before interpreting the POS 1-gram effect (positive estimate on a
  distance), check its correlation with target length and with the length
  difference between target and alternatives.

Conclusion to state in the report: whether the CLASP nulls follow from how
inference is done (G1, G2, G3) or from limited precision (G4, G5).

## Deliverables (all under `$AUDIT`, none in the repo)

- `AUDIT.md`: table with columns
  `ID | stage | file:function | severity | evidence | suggested fix (not applied)`.
  Severity: **blocker** = wrong pairing of data points or a meaningless metric;
  **major** = biased estimates or standard errors; **minor** = hygiene.
- `checks/*.py`, `checks/*.R`, `copies/`, `out/`, `PATHS.md`,
  `git_status_start.txt`, `git_status_end.txt`.
- Order of work: I1–I8 → E → F1–F2 → G → F3–F14 → B → C → D → A.
  Stop after E, F1–F2 and G and report, since they decide whether the current
  numbers can be used.
