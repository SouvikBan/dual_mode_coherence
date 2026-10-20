#!/usr/bin/env Rscript
# lead_lag.R -- Appendix A, "Position": the measures at the preceding (lags 1 to 5) and following
# (leads 1 to 3) utterances of the same story, Natural Stories.
#
# A measure's values at the following utterances cannot affect the current reading time; if they
# predict it as well as the preceding values do, the effect is a slow drift over the story. Rows:
# utterances with five predecessors and three successors in their story. Models per measure:
# current value; + lags; + leads; + both. Also the within-story autocorrelation of each measure.
#
# Output: results/appendix_lead_lag_per_strategy.csv, results/appendix_lead_lag.csv
# Usage:  NCORES=40 Rscript lead_lag.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

ds <- load_dataset("naturalstories")
MEASURES <- c("d_est", "n_mentioned_established", "beta", "n_new", "d_cos")
LAGS <- 1:5
LEADS <- 1:3
shift_names <- function(measure) c(paste0(measure, "_lag", LAGS), paste0(measure, "_lead", LEADS))

# the measures of the neighbouring utterances, per story
shifted_measures <- function(strategy) {
  shifted <- ds$measures %>%
    filter(strategy == !!strategy) %>%
    select(item, sentence_id, all_of(MEASURES)) %>%
    arrange(item, sentence_id) %>%
    group_by(item)
  for (k in LAGS) {
    shifted <- shifted %>%
      mutate(across(all_of(MEASURES), ~ .x[match(sentence_id - k, sentence_id)], .names = paste0("{.col}_lag", k)))
  }
  for (k in LEADS) {
    shifted <- shifted %>%
      mutate(across(all_of(MEASURES), ~ .x[match(sentence_id + k, sentence_id)], .names = paste0("{.col}_lead", k)))
  }
  shifted %>%
    ungroup() %>%
    filter(if_all(matches("_lag[0-9]$|_lead[0-9]$"), ~ !is.na(.x)))
}

autocorrelation <- shifted_measures(DISPLAY_STRATEGY)
cat("within-story autocorrelation at lags 1 to 5 (temperature 1.25):\n")
for (measure in MEASURES) {
  r <- sapply(LAGS, function(k) cor(autocorrelation[[measure]], autocorrelation[[paste0(measure, "_lag", k)]]))
  cat(sprintf("  %-26s %s\n", measure, paste(round(r, 2), collapse = " ")))
}

data_by_strategy <- setNames(lapply(STRATEGIES, function(strategy) {
  shifts <- shifted_measures(strategy) %>% select(item, sentence_id, matches("_lag[0-9]$|_lead[0-9]$"))
  prepare_data(ds, strategy) %>%
    inner_join(shifts, by = c("item", "sentence_id")) %>%
    mutate(across(all_of(c("n_mentioned_established", unlist(lapply(MEASURES, shift_names)))), standardise, .names = "{.col}_s"))
}), STRATEGIES)
cat("rows", nrow(data_by_strategy[[1]]), "utterances", n_distinct(paste(data_by_strategy[[1]]$item, data_by_strategy[[1]]$sentence_id)), "\n")

model_specs <- list(list(label = "controls", measure = NA, terms = NULL))
for (measure in MEASURES) {
  current <- scaled(measure)
  lags <- scaled(paste0(measure, "_lag", LAGS))
  leads <- scaled(paste0(measure, "_lead", LEADS))
  model_specs <- c(model_specs, list(
    list(label = paste("current:", measure), measure = measure, terms = current),
    list(label = paste("current + lags:", measure), measure = measure, terms = c(current, lags)),
    list(label = paste("current + leads:", measure), measure = measure, terms = c(current, leads)),
    list(label = paste("current + lags + leads:", measure), measure = measure, terms = c(current, lags, leads))
  ))
}

jobs <- expand.grid(strategy = STRATEGIES, model = seq_along(model_specs), stringsAsFactors = FALSE)
fit_job <- function(j) {
  strategy <- jobs$strategy[j]
  spec <- model_specs[[jobs$model[j]]]
  model <- fit_model(model_formula(spec$terms), data_by_strategy[[strategy]], "naturalstories")
  coefficients <- fixed_effects(model, "naturalstories")
  pick <- function(name, column) if (!is.na(spec$measure) && name %in% rownames(coefficients)) coefficients[name, column] else NA
  row <- data.frame(model = spec$label, measure = spec$measure, strategy = strategy, aic = AIC(model),
                    estimate = pick(scaled(spec$measure), 1), stat = pick(scaled(spec$measure), 3))
  for (k in LAGS) {
    row[[paste0("lag", k)]] <- pick(scaled(paste0(spec$measure, "_lag", k)), 1)
    row[[paste0("stat_lag", k)]] <- pick(scaled(paste0(spec$measure, "_lag", k)), 3)
  }
  for (k in LEADS) {
    row[[paste0("lead", k)]] <- pick(scaled(paste0(spec$measure, "_lead", k)), 1)
    row[[paste0("stat_lead", k)]] <- pick(scaled(paste0(spec$measure, "_lead", k)), 3)
  }
  row
}
results <- bind_rows(mclapply(seq_len(nrow(jobs)), function(j) {
  tryCatch(fit_job(j), error = function(e) { cat("ERROR", j, conditionMessage(e), "\n", file = stderr()); NULL })
}, mc.cores = N_CORES)) %>%
  group_by(strategy) %>%
  mutate(aic_gain = aic[model == "controls"] - aic) %>%
  ungroup() %>%
  filter(model != "controls")
write_csv(results, file.path(RESULTS_DIR, "appendix_lead_lag_per_strategy.csv"))

significant_counts <- results %>%
  group_by(model) %>%
  summarise(across(c(stat, starts_with("stat_lag"), starts_with("stat_lead")), ~ sum(abs(.x) > 1.96), .names = "n_sig_{.col}"),
            aic_gain_mean = round(mean(aic_gain), 1), .groups = "drop")
summary_table <- results %>%
  filter(strategy == DISPLAY_STRATEGY) %>%
  select(model, aic_gain, estimate, stat, starts_with("lag"), starts_with("lead")) %>%
  mutate(across(where(is.numeric), ~ round(.x, 1))) %>%
  inner_join(significant_counts, by = "model")
write_csv(summary_table, file.path(RESULTS_DIR, "appendix_lead_lag.csv"))
print_table(summary_table %>% select(model, aic_gain, estimate, stat, n_sig_stat, lag1, lag3, lag5, n_sig_stat_lag3, lead1, lead2, lead3, n_sig_stat_lead2, n_sig_stat_lead3),
            "Natural Stories | ms per SD at temperature 1.25; n_sig_* = strategies (of 11) with |t| > 1.96")
cat("finished\n")
