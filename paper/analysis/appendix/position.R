#!/usr/bin/env Rscript
# position.R -- Appendix A, "Position" and "Mentions in the context" (Natural Stories).
#
# Descriptives: reading time and the number of established entities by quarter of the story; the
# correlation of the context-mention control with position. Then every entity measure, the linking
# cost and the number of new entities alone, under the paper's controls (linear position) and under
# the controls plus log position, squared position and the log number of established entities;
# all 11 strategies.
#
# Output: results/appendix_position_per_strategy.csv, results/appendix_position.csv
# Usage:  NCORES=40 Rscript position.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

ds <- load_dataset("naturalstories")
MEASURES <- c(ENTITY_MEASURES, "beta", "n_new")

add_nonlinear_terms <- function(data) {
  data %>% mutate(log_position_s = standardise(log(position)),
                  position_squared_s = standardise(position^2),
                  log_established_s = standardise(log1p(n_established)))
}
data_by_strategy <- setNames(lapply(STRATEGIES, function(strategy) add_nonlinear_terms(prepare_data(ds, strategy))), STRATEGIES)

# ---- descriptives (temperature 1.25 rows; the measures do not enter) ------------------------------
# reading time per word read, averaged over the rows of each quarter of a story
display <- data_by_strategy[[DISPLAY_STRATEGY]]
quarters <- display %>%
  group_by(item) %>%
  mutate(quarter = ceiling(4 * position / max(position))) %>%
  group_by(quarter) %>%
  summarise(mean_reading_time_per_word = mean(y / length), mean_utterance_reading_time = mean(y),
            mean_established = mean(n_established), .groups = "drop")
print_table(quarters, "reading time and established entities by quarter of the story")
cat(sprintf("reading time per word, last vs first quarter: %+.1f%%; established entities: x%.1f\n",
            100 * (quarters$mean_reading_time_per_word[4] / quarters$mean_reading_time_per_word[1] - 1),
            quarters$mean_established[4] / quarters$mean_established[1]))
utterances <- display %>% distinct(item, sentence_id, .keep_all = TRUE)
cat(sprintf("r(mentions in the context, position) = %.3f; r(mentions in the context, established entities) = %.3f\n",
            cor(utterances$ctx_mentions, utterances$position), cor(utterances$ctx_mentions, utterances$n_established)))

# ---- each measure alone under the two control sets ----------------------------------------------
CONTROL_SETS <- c(linear = CONTROLS,
                  nonlinear = paste(CONTROLS, "+ log_position_s + position_squared_s + log_established_s"))
results <- bind_rows(lapply(names(CONTROL_SETS), function(control_set) {
  models <- c(list(control_model_spec()),
              lapply(MEASURES, function(m) list(label = m, terms = scaled(m), term = scaled(m), heldout = FALSE)))
  compare_models(ds, data_by_strategy, models, controls = CONTROL_SETS[[control_set]]) %>%
    mutate(controls = control_set)
}))
write_csv(results, file.path(RESULTS_DIR, "appendix_position_per_strategy.csv"))

n_positive <- results %>%
  group_by(controls, model) %>%
  summarise(n_positive = sum(estimate > 0), .groups = "drop")
summary_table <- summarise_comparison(results, by = c("controls", "model")) %>%
  select(-dll_x100, -folds_positive) %>%
  inner_join(n_positive, by = c("controls", "model")) %>%
  arrange(model, controls)
write_csv(summary_table, file.path(RESULTS_DIR, "appendix_position.csv"))
print_table(summary_table, "each measure alone | ms per SD at temperature 1.25 (t); mean; strategies significant and positive; AIC gain")
cat("finished\n")
