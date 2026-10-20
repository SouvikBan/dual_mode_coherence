#!/usr/bin/env Rscript
# established_count.R -- Appendix A, "Mentions in the context", and Section 5 (ii): the number of
# established entities an utterance mentions (Natural Stories).
#
# Models: the count alone (with leave-one-story-out held-out); d_est alone and given the count;
# d_part and the history version of d_part alone, given the count, and given the count and d_est.
#
# Output: results/appendix_established_count_per_strategy.csv, results/appendix_established_count.csv
# Usage:  NCORES=40 Rscript established_count.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

ds <- load_dataset("naturalstories")
EXTRA <- c("d_part_history", "d_part_history_A")
data_by_strategy <- setNames(lapply(STRATEGIES, function(strategy) {
  prepare_data(ds, strategy, extra_columns = EXTRA) %>%
    mutate(across(all_of(c("n_mentioned_established", EXTRA)), standardise, .names = "{.col}_s"))
}), STRATEGIES)

utterances <- data_by_strategy[[DISPLAY_STRATEGY]] %>% distinct(item, sentence_id, .keep_all = TRUE)
cat(sprintf("r(count, d_est) = %.2f; r(count, established entities) = %.2f; r(count, position) = %.2f; r(d_part, d_part_history) = %.2f\n",
            cor(utterances$n_mentioned_established, utterances$d_est), cor(utterances$n_mentioned_established, utterances$n_established),
            cor(utterances$n_mentioned_established, utterances$position), cor(utterances$d_part, utterances$d_part_history)))

spec <- function(label, terms, term, heldout = FALSE) list(label = label, terms = terms, term = term, heldout = heldout)
COUNT <- "n_mentioned_established_s"
models <- list(
  control_model_spec(),
  spec("count alone", COUNT, COUNT, heldout = TRUE),
  spec("d_est alone", "d_est_s", "d_est_s", heldout = TRUE),
  spec("d_est | count", c(COUNT, "d_est_s"), "d_est_s"),
  spec("count | d_est", c(COUNT, "d_est_s"), COUNT),
  spec("d_part alone", "d_part_s", "d_part_s", heldout = TRUE),
  spec("d_part | count", c(COUNT, "d_part_s"), "d_part_s"),
  spec("d_part | count + d_est", c(COUNT, "d_est_s", "d_part_s"), "d_part_s"),
  spec("d_part_A alone", "d_part_A_s", "d_part_A_s", heldout = TRUE),
  spec("d_part_A | count + d_est", c(COUNT, "d_est_s", "d_part_A_s"), "d_part_A_s"),
  spec("d_part_history alone", "d_part_history_s", "d_part_history_s", heldout = TRUE),
  spec("d_part_history | count + d_est", c(COUNT, "d_est_s", "d_part_history_s"), "d_part_history_s"),
  spec("d_part_history_A alone", "d_part_history_A_s", "d_part_history_A_s", heldout = TRUE),
  spec("d_part_history_A | count + d_est", c(COUNT, "d_est_s", "d_part_history_A_s"), "d_part_history_A_s")
)
results <- compare_models(ds, data_by_strategy, models)
write_csv(results, file.path(RESULTS_DIR, "appendix_established_count_per_strategy.csv"))
summary_table <- summarise_comparison(results)
write_csv(summary_table, file.path(RESULTS_DIR, "appendix_established_count.csv"))
print_table(summary_table, "Natural Stories | ms per SD at temperature 1.25 (t); mean; strategies significant; AIC gain; held-out gain x100 (folds > 0)")
cat("finished\n")
