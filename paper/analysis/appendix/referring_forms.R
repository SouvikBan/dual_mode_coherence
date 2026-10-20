#!/usr/bin/env Rscript
# referring_forms.R -- Appendix A, "Claims tested and not reported above": a distance over referring forms.
#
# The information value of the target's referring forms under two orderings of forms, the Givenness
# Hierarchy and the typical antecedent distance fitted on GUM (measures/referring_forms.py), each
# added alone to the controls: over all established entities the target mentions (rows where both
# are defined) and for the first referent only; held-out gains on unseen text; all 11 strategies.
#
# Output: results/appendix_referring_forms_<dataset>_per_strategy.csv, results/appendix_referring_forms_<dataset>.csv
# Usage:  NCORES=40 Rscript referring_forms.R <naturalstories|clasp>

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

dataset <- script_arguments()[1]
ds <- load_dataset(dataset)
SETS <- list(
  `all established entities` = c("d_form_accessibility", "d_form_givenness"),
  `first referent` = c("d_form_first_accessibility", "d_form_first_givenness")
)
if (!all(unlist(SETS) %in% names(ds$measures))) stop("run measures/referring_forms.py first")

spec <- function(label, terms, term) list(label = label, terms = terms, term = term, heldout = TRUE)
results <- bind_rows(lapply(names(SETS), function(set) {
  variables <- SETS[[set]]
  data_by_strategy <- setNames(lapply(STRATEGIES, function(strategy) {
    prepare_data(ds, strategy, extra_columns = variables) %>%
      filter(if_all(all_of(variables), ~ !is.na(.x))) %>%
      mutate(across(all_of(variables), standardise, .names = "{.col}_s"))
  }), STRATEGIES)
  models <- c(list(control_model_spec()),
              lapply(variables, function(v) spec(paste(v, "alone"), scaled(v), scaled(v))))
  compare_models(ds, data_by_strategy, models) %>% mutate(rows = set)
}))
write_csv(results, file.path(RESULTS_DIR, paste0("appendix_referring_forms_", dataset, "_per_strategy.csv")))
summary_table <- summarise_comparison(results, by = c("rows", "model"))
write_csv(summary_table, file.path(RESULTS_DIR, paste0("appendix_referring_forms_", dataset, ".csv")))
print_table(summary_table, paste(dataset, "| referring-form distances alone | per SD at temperature 1.25 (stat); mean; strategies significant; AIC; held-out gain x100"))
cat("finished\n")
