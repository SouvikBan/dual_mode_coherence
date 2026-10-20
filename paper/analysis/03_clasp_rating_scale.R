#!/usr/bin/env Rscript
# 03_clasp_rating_scale.R -- CLASP effects in rating points per SD (paper, Section 4).
#
# The ordered beta model estimates effects in logits. The paper reports them in rating points on
# the 1 to 4 scale: each logit coefficient is multiplied by the average slope of the expected
# rating with respect to the linear predictor, 3 * mean_i dE[y]/d(eta_i), with eta_i the full
# linear predictor of rating i (fixed effects and the three random intercepts) and E[y] the
# ordered-beta expectation including the point masses at 0 and 1. The factor is computed from the
# control model (and, as a check, from the full model) of every strategy; it is about 0.81.
#
# Input:  results/estimates_clasp.csv (logits, from 02_estimates.R)
# Output: results/clasp_rating_scale_factors.csv
#         results/estimates_clasp_rating_points.csv and _summary.csv (estimate and SE converted;
#         statistics and AIC gains unchanged)
# Usage: NCORES=11 Rscript 03_clasp_rating_scale.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

ds <- load_dataset("clasp")
RATING_RANGE <- 3   # 4 - 1

conversion_factor <- function(model, label, strategy) {
  cutpoints <- ordbeta_cutpoints(model)
  eta_conditional <- predict(model, type = "link")
  eta_fixed <- predict(model, type = "link", re.form = NA)
  shift <- 0.04   # finite-difference check over a shift the size of the largest CLASP effect
  data.frame(
    strategy = strategy,
    model = label,
    factor_ame_cond = RATING_RANGE * mean(ordbeta_expected_slope(eta_conditional, cutpoints)),
    factor_ame_fixed = RATING_RANGE * mean(ordbeta_expected_slope(eta_fixed, cutpoints)),
    factor_fd_004 = RATING_RANGE * mean(ordbeta_expected(eta_conditional + shift, cutpoints) -
                                        ordbeta_expected(eta_conditional, cutpoints)) / shift,
    mean_rating = 1 + RATING_RANGE * mean(ordbeta_expected(eta_conditional, cutpoints)),
    cutpoint_1 = cutpoints[1],
    cutpoint_2 = cutpoints[2]
  )
}

factors_for_strategy <- function(strategy) {
  data <- prepare_data(ds, strategy, "all")
  control_model <- fit_model(model_formula(), data, "clasp")
  full_model <- fit_model(model_formula(scaled(FULL_MODEL_MEASURES)), data, "clasp")
  cat("done", strategy, "\n", file = stderr())
  bind_rows(conversion_factor(control_model, "controls", strategy),
            conversion_factor(full_model, "full", strategy))
}

factors <- bind_rows(mclapply(STRATEGIES, function(strategy) {
  tryCatch(factors_for_strategy(strategy),
           error = function(e) { cat("ERROR", strategy, conditionMessage(e), "\n", file = stderr()); NULL })
}, mc.cores = N_CORES))
write_csv(factors, file.path(RESULTS_DIR, "clasp_rating_scale_factors.csv"))
print_table(factors %>% mutate(across(where(is.numeric), ~ signif(.x, 4))), "rating points per logit")

# ---- convert the estimates ---------------------------------------------------------------------
factor_by_strategy <- factors %>% filter(model == "controls") %>% select(strategy, factor = factor_ame_cond)
cat(sprintf("\ntemperature 1.25: one logit = %.4f rating points at the average rating\n",
            factor_by_strategy$factor[factor_by_strategy$strategy == DISPLAY_STRATEGY]))

estimates <- read_csv(file.path(RESULTS_DIR, "estimates_clasp.csv"), show_col_types = FALSE) %>%
  inner_join(factor_by_strategy, by = "strategy") %>%
  mutate(estimate = estimate * factor, se = se * factor) %>%
  select(-factor)
write_csv(estimates, file.path(RESULTS_DIR, "estimates_clasp_rating_points.csv"))

summary_table <- estimates %>%
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
write_csv(summary_table, file.path(RESULTS_DIR, "estimates_clasp_rating_points_summary.csv"))

for (subset in ds$subsets) {
  print_table(summary_table %>% filter(subset == !!subset, model == "alone") %>% arrange(-aic_gain_t125),
              paste("CLASP", subset, "| each measure alone | rating points per SD at temperature 1.25"))
  print_table(summary_table %>% filter(subset == !!subset, model == "full"),
              paste("CLASP", subset, "| full model | rating points per SD"))
}
cat("finished\n")
