#!/usr/bin/env Rscript
# next_mention.R -- Appendix A, "Claims tested and not reported above", and Section 5 (i), Natural Stories.
#
# (a) First referent. On the targets whose first referent is an entity of the previous utterance, the
#     surprisal of that referent under each next-mention account fitted on GUM (grammatical salience,
#     recency, frequency, information status, and all four) is added alone to the controls, and given
#     the uniform expectation log2 of the number of candidates; leave-one-story-out held-out gains and
#     the paired per-story difference of each account against the uniform expectation. These measures
#     do not depend on the alternatives, so they are fitted once (temperature 1.25 rows).
# (b) Pronoun interpretation. On the targets that begin with a personal pronoun of an established
#     entity, the Expectancy, Mirror and Bayesian surprisals (from the alternatives of each strategy)
#     and the uniform log2 of the number of agreement-compatible candidates, alone; held-out gains and
#     paired per-story differences between the models, over the 11 strategies.
#
# Output: results/appendix_next_mention_first_referent.csv, results/appendix_next_mention_pronoun*.csv
# Usage:  NCORES=11 Rscript next_mention.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

ds <- load_dataset("naturalstories")
FIRST_COLUMNS <- c("log2_M", "first_in_previous", paste0("surprisal_first_", NEXT_MENTION_ACCOUNTS))
PRONOUN_COLUMNS <- c("pronoun_eligible", "n_candidates", "surprisal_uniform", "surprisal_expectancy", "surprisal_mirror", "surprisal_bayesian")
if (!all(c(FIRST_COLUMNS, PRONOUN_COLUMNS) %in% names(ds$measures))) stop("run measures/next_mention.py first")

# One model on one row set: estimate per SD of `term`, AIC gain over the controls on the same rows,
# per-story held-out log-likelihoods of the model and of the controls.
fit_on_rows <- function(data, label, terms, term) {
  controls <- fit_model(model_formula(), data, "naturalstories")
  model <- fit_model(model_formula(terms), data, "naturalstories")
  coefficients <- fixed_effects(model, "naturalstories")
  list(
    row = data.frame(model = label, n_rows = nrow(data), n_targets = n_distinct(data$sentence_key),
                     estimate = coefficients[term, 1], stat = coefficients[term, 3],
                     aic_gain = AIC(controls) - AIC(model)),
    fold_ll = fold_log_likelihoods(ds, model_formula(terms), data),
    fold_ll_controls = fold_log_likelihoods(ds, model_formula(), data)
  )
}

summarise_fits <- function(fits) {
  bind_rows(lapply(fits, function(fit) {
    difference <- fit$fold_ll - fit$fold_ll_controls
    fit$row %>% mutate(dll_x100 = 100 * mean(difference), folds_positive = sum(difference > 0))
  }))
}

paired_difference <- function(fits, a, b) {
  difference <- fits[[a]]$fold_ll - fits[[b]]$fold_ll
  data.frame(comparison = paste(a, "-", b), dll_x100 = 100 * mean(difference),
             min_x100 = 100 * min(difference), max_x100 = 100 * max(difference), folds_positive = sum(difference > 0))
}

# ---- (a) first referent -----------------------------------------------------------------------
first_rows <- prepare_data(ds, DISPLAY_STRATEGY, extra_columns = FIRST_COLUMNS) %>%
  filter(first_in_previous == 1, !is.na(surprisal_first_grammatical), !is.na(log2_M)) %>%
  mutate(across(all_of(c("log2_M", paste0("surprisal_first_", NEXT_MENTION_ACCOUNTS))), standardise, .names = "{.col}_s"))
cat("first referent in the previous utterance:", n_distinct(first_rows$sentence_key), "targets,", nrow(first_rows), "observations\n")

first_fits <- list(uniform = fit_on_rows(first_rows, "uniform (log2 candidates)", "log2_M_s", "log2_M_s"))
for (account in NEXT_MENTION_ACCOUNTS) {
  variable <- scaled(paste0("surprisal_first_", account))
  first_fits[[account]] <- fit_on_rows(first_rows, paste(account, "alone"), variable, variable)
  first_fits[[paste(account, "| uniform")]] <- fit_on_rows(first_rows, paste(account, "| uniform"), c("log2_M_s", variable), variable)
}
first_summary <- summarise_fits(first_fits)
first_pairs <- bind_rows(lapply(NEXT_MENTION_ACCOUNTS, function(account) paired_difference(first_fits, account, "uniform")))
write_csv(first_summary, file.path(RESULTS_DIR, "appendix_next_mention_first_referent.csv"))
write_csv(first_pairs, file.path(RESULTS_DIR, "appendix_next_mention_first_referent_paired.csv"))
print_table(first_summary %>% mutate(across(where(is.numeric), ~ signif(.x, 3))),
            "first referent | ms per SD (t); AIC gain; held-out gain x100 over the controls (stories > 0)")
print_table(first_pairs %>% mutate(across(where(is.numeric), ~ signif(.x, 3))),
            "first referent | paired held-out difference against the uniform expectation (x100 nats per observation, over stories)")

# ---- (b) pronoun interpretation -----------------------------------------------------------------
PRONOUN_MODELS <- c(uniform = "surprisal_uniform", expectancy = "surprisal_expectancy", mirror = "surprisal_mirror", bayesian = "surprisal_bayesian")
pronoun_strategy <- function(strategy) {
  data <- prepare_data(ds, strategy, extra_columns = PRONOUN_COLUMNS) %>%
    filter(pronoun_eligible == 1) %>%
    mutate(across(all_of(unname(PRONOUN_MODELS)), standardise, .names = "{.col}_s"))
  fits <- lapply(PRONOUN_MODELS, function(variable) fit_on_rows(data, variable, scaled(variable), scaled(variable)))
  names(fits) <- names(PRONOUN_MODELS)
  summary <- summarise_fits(fits) %>% mutate(model = names(PRONOUN_MODELS), strategy = strategy)
  pairs <- bind_rows(paired_difference(fits, "bayesian", "expectancy"), paired_difference(fits, "bayesian", "mirror"),
                     paired_difference(fits, "bayesian", "uniform")) %>% mutate(strategy = strategy)
  cat("done", strategy, "\n", file = stderr())
  list(summary = summary, pairs = pairs)
}
pronoun <- mclapply(STRATEGIES, function(strategy) {
  tryCatch(pronoun_strategy(strategy), error = function(e) { cat("ERROR", strategy, conditionMessage(e), "\n", file = stderr()); NULL })
}, mc.cores = N_CORES)
pronoun_summary <- bind_rows(lapply(pronoun, `[[`, "summary"))
pronoun_pairs <- bind_rows(lapply(pronoun, `[[`, "pairs"))
write_csv(pronoun_summary, file.path(RESULTS_DIR, "appendix_next_mention_pronoun_per_strategy.csv"))
write_csv(pronoun_pairs, file.path(RESULTS_DIR, "appendix_next_mention_pronoun_paired_per_strategy.csv"))
cat("\npronoun targets:", pronoun_summary$n_targets[1], "\n")
pronoun_table <- pronoun_summary %>%
  group_by(model) %>%
  summarise(n_targets = first(n_targets),
            estimate_t125 = signif(first(estimate[strategy == DISPLAY_STRATEGY]), 3), stat_t125 = round(first(stat[strategy == DISPLAY_STRATEGY]), 1),
            estimate_mean = signif(mean(estimate), 3), n_significant = sum(abs(stat) > 1.96), n_strategies = n(),
            aic_gain_t125 = round(first(aic_gain[strategy == DISPLAY_STRATEGY]), 1), aic_gain_mean = round(mean(aic_gain), 1),
            dll_x100 = round(mean(dll_x100), 2), folds_positive = round(mean(folds_positive), 1), .groups = "drop")
pairs_table <- pronoun_pairs %>%
  group_by(comparison) %>%
  summarise(dll_x100_t125 = round(first(dll_x100[strategy == DISPLAY_STRATEGY]), 3), dll_x100_mean = round(mean(dll_x100), 3),
            min_x100_t125 = round(first(min_x100[strategy == DISPLAY_STRATEGY]), 2), max_x100_t125 = round(first(max_x100[strategy == DISPLAY_STRATEGY]), 2),
            strategies_positive = sum(dll_x100 > 0), .groups = "drop")
write_csv(pronoun_table, file.path(RESULTS_DIR, "appendix_next_mention_pronoun.csv"))
write_csv(pairs_table, file.path(RESULTS_DIR, "appendix_next_mention_pronoun_paired.csv"))
print_table(pronoun_table, "pronoun interpretation | ms per SD at temperature 1.25 (t); mean; strategies significant; AIC; held-out gain x100")
print_table(pairs_table, "pronoun interpretation | paired held-out differences between models (x100 nats per observation)")
cat("finished\n")
