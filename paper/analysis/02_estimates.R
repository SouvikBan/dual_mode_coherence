#!/usr/bin/env Rscript
# 02_estimates.R -- the effect sizes of the paper (Figure 1, Figure 2, Figure 3, Table 1).
#
# For every sampling strategy: the control model; each measure added alone to the controls
# (the 15 measures of the full model and the four alone-only measures); and the full model with
# all 15 measures. CLASP is fitted on all 100 contexts and on the 80 contexts with established
# entities. One fit per model on all data; estimates are per SD of the predictor.
#
# Output: results/estimates_<dataset>.csv          one row per strategy, subset, model and term
#         results/estimates_<dataset>_summary.csv  per subset, model and term: estimate, SE and
#                                                  statistic at temperature 1.25, mean estimate,
#                                                  strategies with |stat| > 1.96, AIC gain
# CLASP estimates are logits per SD here; 03_clasp_rating_scale.R converts them to rating points.
# Usage: NCORES=11 Rscript 02_estimates.R <naturalstories|clasp>

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

dataset <- script_arguments()[1]
stopifnot(dataset %in% c("naturalstories", "clasp"))
ds <- load_dataset(dataset)

# Every measure alone and the full model, for one strategy and subset.
estimates_for <- function(strategy, subset) {
  data <- prepare_data(ds, strategy, subset)
  control_model <- fit_model(model_formula(), data, dataset)
  rows <- list()
  add_row <- function(model_label, term, coefficients, aic_gain) {
    rows[[length(rows) + 1]] <<- data.frame(subset = subset, model = model_label, term = term,
                                            coefficients, aic_gain = aic_gain, strategy = strategy)
  }
  for (measure in ALL_MEASURES) {
    if (sd(data[[measure]]) == 0) next
    model <- fit_model(model_formula(scaled(measure)), data, dataset)
    add_row("alone", measure, coefficient_row(model, dataset, scaled(measure)), AIC(control_model) - AIC(model))
  }
  full_model <- fit_model(model_formula(scaled(FULL_MODEL_MEASURES)), data, dataset)
  for (variable in c(CONTROL_VARIABLES, FULL_MODEL_MEASURES)) {
    add_row("full", variable, coefficient_row(full_model, dataset, scaled(variable)), AIC(control_model) - AIC(full_model))
  }
  bind_rows(rows)
}

estimates_for_strategy <- function(strategy) {
  cache_file <- file.path(CACHE_DIR, paste0("estimates_", dataset, "_", strategy, ".rds"))
  if (file.exists(cache_file)) return(readRDS(cache_file))
  result <- bind_rows(lapply(ds$subsets, function(subset) estimates_for(strategy, subset)))
  saveRDS(result, cache_file)
  cat("done", strategy, "\n", file = stderr())
  result
}

estimates <- bind_rows(mclapply(STRATEGIES, function(strategy) {
  tryCatch(estimates_for_strategy(strategy),
           error = function(e) { cat("ERROR", strategy, conditionMessage(e), "\n", file = stderr()); NULL })
}, mc.cores = N_CORES))
write_csv(estimates, file.path(RESULTS_DIR, paste0("estimates_", dataset, ".csv")))

summarise_estimates <- function(estimates) {
  estimates %>%
    group_by(subset, model, term) %>%
    summarise(
      estimate_t125 = signif(first(estimate[strategy == DISPLAY_STRATEGY], default = NA), 3),
      se_t125 = signif(first(se[strategy == DISPLAY_STRATEGY], default = NA), 2),
      stat_t125 = round(first(stat[strategy == DISPLAY_STRATEGY], default = NA), 1),
      estimate_mean = signif(mean(estimate), 3),
      n_significant = sum(abs(stat) > 1.96),
      n_strategies = n(),
      aic_gain_t125 = round(first(aic_gain[strategy == DISPLAY_STRATEGY], default = NA), 1),
      aic_gain_mean = round(mean(aic_gain), 1),
      .groups = "drop"
    )
}

summary_table <- summarise_estimates(estimates)
write_csv(summary_table, file.path(RESULTS_DIR, paste0("estimates_", dataset, "_summary.csv")))

unit <- if (dataset == "naturalstories") "ms per SD" else "logits per SD"
for (subset in ds$subsets) {
  print_table(summary_table %>% filter(subset == !!subset, model == "alone") %>% arrange(-aic_gain_t125),
              paste(dataset, subset, "| each measure alone |", unit, "at temperature 1.25"))
  print_table(summary_table %>% filter(subset == !!subset, model == "full"),
              paste(dataset, subset, "| full model (controls + 15 measures)"))
}
cat("finished\n")
