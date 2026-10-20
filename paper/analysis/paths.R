# paths.R -- every input the analyses read and every place they write to.
# Edit the "external inputs" block for a new machine; everything else is relative to paper/.

# ---- external inputs -------------------------------------------------------------------------
# Output of the alternative-generation and annotation pipeline (scripts/entity_iv, stages 1 to 4).
DATA_ROOT <- Sys.getenv("ENTITY_IV_DATA",
                        "/hits/fast/nlp/banerjsk/crac2026-corpipe/corpipe_seeded_smoke/vllm_raw_generation")
ANNOTATED_ALTERNATIVES <- file.path(DATA_ROOT, "annotated_alternatives")   # <dataset>/<document>.json

# Natural Stories: self-paced reading times (Futrell et al., 2021) and the word-level surprisal
# file of the repository (Pythia-70m-deduped, step 1,000; scripts/score_stories.py).
NS_READING_TIMES <- file.path(DATA_ROOT, "naturalstories/naturalstories_RTS/processed_RTs.tsv")
NS_WORDS <- file.path(DATA_ROOT, "naturalstories/naturalstories_RTS/all_stories.tok")
# The repository copy (annotated/surprisal/) has a malformed six-name header over five columns and
# rounded values; the analyses use the pipeline's copy next to DATA_ROOT.
NS_WORD_SURPRISAL <- Sys.getenv("NS_WORD_SURPRISAL",
                                file.path(dirname(DATA_ROOT), "annotated/surprisal/pythia-70m-deduped-step1000-full.tsv"))
NS_GOLD_COUNTS <- file.path(DATA_ROOT, "gold_counts/naturalstories_entity_counts.csv")

# CLASP: one row per rating with the sentence surprisal of the version under Pythia-12b-deduped
# (scripts/score_clasp.py on BLL2018/data/clasp_with_context_by_worker_and_sent.csv, which
# scripts/preprocess_clasp.Rmd writes); the same directory holds the other Pythia sizes for
# the surprisal-size appendix.
CLASP_SURPRISAL_DIR <- Sys.getenv("CLASP_SURPRISAL_DIR",
  "/hits/fast/nlp/banerjsk/dual_mode_coherence_audit/scratch_run/annotated/surprisal/clasp/clasp_with_context_by_worker_and_sent")
CLASP_RATINGS <- file.path(CLASP_SURPRISAL_DIR, "pythia-12b-deduped-full.tsv")
CLASP_POSITIONS <- file.path(DATA_ROOT, "iv_values/clasp/iv_temp_125.csv")   # sentence_index per version
CLASP_GOLD_COUNTS <- file.path(DATA_ROOT, "gold_counts/clasp_entity_counts.csv")

# GUM (Zeldes, 2017) dependency files with the salience annotation, for measures/train_salience.py.
GUM_DIR <- Sys.getenv("GUM_DIR", "/hits/fast/nlp/banerjsk/gum/dep")

# ---- inside paper/ ----------------------------------------------------------------------------
DATA_DIR <- file.path(PAPER_DIR, "data")             # measures/<measure>/<dataset>/*.csv and word-level files
MEASURES_DIR <- Sys.getenv("PAPER_MEASURES_DIR", file.path(DATA_DIR, "measures"))
RESULTS_DIR <- file.path(PAPER_DIR, "results")
TABLES_DIR <- file.path(RESULTS_DIR, "tables")
FIGURES_DIR <- file.path(PAPER_DIR, "figures")
CACHE_DIR <- file.path(RESULTS_DIR, "cache")

NS_SENTENCE_TABLE <- file.path(DATA_DIR, "naturalstories_reading_times.csv")   # written by 01_prepare_naturalstories.R
NS_FIRST_MENTION_ZONES <- file.path(DATA_DIR, "naturalstories_first_mention_zones.csv")
NS_WORD_SURPRISAL_RECOMPUTED <- file.path(DATA_DIR, "naturalstories_word_surprisal.csv")
NS_WORD_ZIPF <- file.path(DATA_DIR, "naturalstories_word_zipf.csv")

for (dir in c(DATA_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, CACHE_DIR)) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
}
