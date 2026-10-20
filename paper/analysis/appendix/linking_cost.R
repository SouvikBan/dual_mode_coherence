#!/usr/bin/env Rscript
# linking_cost.R -- Appendix A, "Linking cost", and Section 5 (iv).
#
# The linking cost beta alone and given the number of new entities, the semantic distance d_cos,
# or all ten non-entity measures; its information value d_beta alone and given the cost (and the
# cost given it); the anticipatory form alone. Held-out gains (unseen stories / contexts) for the
# models fitted alone. Item-level correlations of the cost with d_cos and the new-entity count.
#
# Output: results/appendix_linking_cost_<dataset>_per_strategy.csv, results/appendix_linking_cost_<dataset>.csv
# Usage:  NCORES=40 Rscript linking_cost.R <naturalstories|clasp>

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

dataset <- script_arguments()[1]
ds <- load_dataset(dataset)
data_by_strategy <- setNames(lapply(STRATEGIES, function(strategy) prepare_data(ds, strategy)), STRATEGIES)

correlations <- bind_rows(lapply(STRATEGIES, function(strategy) {
  items <- data_by_strategy[[strategy]] %>% distinct(across(all_of(measure_keys(dataset))), .keep_all = TRUE)
  data.frame(strategy = strategy, r_beta_cos = cor(items$beta, items$d_cos), r_beta_new = cor(items$beta, items$n_new),
             r_dbeta_beta = cor(items$d_beta, items$beta))
}))
print_table(correlations %>% mutate(across(-strategy, ~ round(.x, 3))), paste(dataset, "| item-level correlations"))

spec <- function(label, terms, term, heldout = FALSE) list(label = label, terms = terms, term = term, heldout = heldout)
NON_ENTITY <- scaled(NON_ENTITY_MEASURES)
models <- list(
  control_model_spec(),
  spec("beta alone", "beta_s", "beta_s", heldout = TRUE),
  spec("n_new alone", "n_new_s", "n_new_s", heldout = TRUE),
  spec("beta | n_new", c("n_new_s", "beta_s"), "beta_s"),
  spec("n_new | beta", c("n_new_s", "beta_s"), "n_new_s"),
  spec("beta | d_cos", c("d_cos_s", "beta_s"), "beta_s"),
  spec("d_cos | beta", c("d_cos_s", "beta_s"), "d_cos_s"),
  spec("beta | all 10 non-entity", c(NON_ENTITY, "beta_s"), "beta_s"),
  spec("d_beta alone", "d_beta_s", "d_beta_s", heldout = TRUE),
  spec("d_beta | beta", c("beta_s", "d_beta_s"), "d_beta_s"),
  spec("beta | d_beta", c("beta_s", "d_beta_s"), "beta_s"),
  spec("d_beta_A alone", "d_beta_A_s", "d_beta_A_s", heldout = TRUE),
  spec("d_est + beta (block)", c("d_est_s", "beta_s"), NULL, heldout = TRUE)
)
results <- compare_models(ds, data_by_strategy, models)
write_csv(results, file.path(RESULTS_DIR, paste0("appendix_linking_cost_", dataset, "_per_strategy.csv")))
summary_table <- summarise_comparison(results)
write_csv(summary_table, file.path(RESULTS_DIR, paste0("appendix_linking_cost_", dataset, ".csv")))
print_table(summary_table, paste(dataset, "| per SD at temperature 1.25 (stat); mean; strategies significant; AIC gain; held-out gain x100 (folds > 0)"))
cat("finished\n")
