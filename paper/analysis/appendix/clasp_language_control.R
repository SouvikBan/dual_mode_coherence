#!/usr/bin/env Rscript
# clasp_language_control.R -- Appendix A, "Language of the CLASP version".
#
# One indicator per translation route (Czech, German, Spanish, French against the English
# original) is added to the controls. Reported: the language coefficients and the AIC gain of the
# four indicators; every measure alone and the full model, with and without the indicators; all
# 11 strategies; on all contexts and on the 80 with established entities. Estimates in logits per
# SD, with the rating-point conversion of 03_clasp_rating_scale.R applied to the summary.
#
# Output: results/appendix_clasp_language_control_per_strategy.csv, results/appendix_clasp_language_control.csv
# Usage:  NCORES=22 Rscript clasp_language_control.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

ds <- load_dataset("clasp")
LANGUAGE_LEVELS <- unname(CLASP_LANGUAGES)
CONTROLS_WITH_LANGUAGE <- paste(CONTROLS, "+ language")

with_language <- function(data) {
  data$language <- factor(sub(".*::", "", data$item_id), levels = LANGUAGE_LEVELS)
  data
}

fit_one <- function(strategy, subset) {
  data <- with_language(prepare_data(ds, strategy, subset))
  rows <- list()
  add_row <- function(model_label, term, coefficients, aic_gain_language = NA) {
    rows[[length(rows) + 1]] <<- data.frame(subset = subset, strategy = strategy, model = model_label, term = term,
                                            coefficients, aic_gain_language = aic_gain_language)
  }
  controls <- fit_model(model_formula(), data, "clasp")
  controls_language <- fit_model(model_formula(controls = CONTROLS_WITH_LANGUAGE), data, "clasp")
  for (language in LANGUAGE_LEVELS[-1]) {
    add_row("controls + language", language, coefficient_row(controls_language, "clasp", paste0("language", language)),
            AIC(controls) - AIC(controls_language))
  }
  for (measure in ALL_MEASURES) {
    if (sd(data[[measure]]) == 0) next
    for (language_control in c(FALSE, TRUE)) {
      model <- fit_model(model_formula(scaled(measure), controls = if (language_control) CONTROLS_WITH_LANGUAGE else CONTROLS), data, "clasp")
      add_row(if (language_control) "alone + language" else "alone", measure, coefficient_row(model, "clasp", scaled(measure)))
    }
  }
  for (language_control in c(FALSE, TRUE)) {
    model <- fit_model(model_formula(scaled(FULL_MODEL_MEASURES), controls = if (language_control) CONTROLS_WITH_LANGUAGE else CONTROLS), data, "clasp")
    for (measure in FULL_MODEL_MEASURES) {
      add_row(if (language_control) "full + language" else "full", measure, coefficient_row(model, "clasp", scaled(measure)))
    }
  }
  cat("done", strategy, subset, "\n", file = stderr())
  bind_rows(rows)
}

jobs <- expand.grid(strategy = STRATEGIES, subset = ds$subsets, stringsAsFactors = FALSE)
results <- bind_rows(mclapply(seq_len(nrow(jobs)), function(j) {
  tryCatch(fit_one(jobs$strategy[j], jobs$subset[j]),
           error = function(e) { cat("ERROR", j, conditionMessage(e), "\n", file = stderr()); NULL })
}, mc.cores = N_CORES))
write_csv(results, file.path(RESULTS_DIR, "appendix_clasp_language_control_per_strategy.csv"))

factors <- read_csv(file.path(RESULTS_DIR, "clasp_rating_scale_factors.csv"), show_col_types = FALSE)
rating_points_per_logit <- factors$factor_ame_cond[factors$model == "controls" & factors$strategy == DISPLAY_STRATEGY]
summary_table <- results %>%
  group_by(subset, model, term) %>%
  summarise(estimate_t125 = signif(first(estimate[strategy == DISPLAY_STRATEGY], default = NA), 3),
            rating_points_t125 = signif(first(estimate[strategy == DISPLAY_STRATEGY], default = NA) * rating_points_per_logit, 3),
            z_t125 = round(first(stat[strategy == DISPLAY_STRATEGY], default = NA), 1),
            estimate_mean = signif(mean(estimate), 3),
            n_significant = sum(abs(stat) > 1.96),
            n_strategies = n(),
            aic_gain_language_t125 = round(first(aic_gain_language[strategy == DISPLAY_STRATEGY], default = NA), 1),
            .groups = "drop")
write_csv(summary_table, file.path(RESULTS_DIR, "appendix_clasp_language_control.csv"))
for (subset in ds$subsets) {
  print_table(summary_table %>% filter(subset == !!subset, model == "controls + language"),
              paste("CLASP", subset, "| language indicators (logits against English; AIC gain of the four)"))
  print_table(summary_table %>% filter(subset == !!subset, model %in% c("alone", "alone + language")) %>% arrange(term, model),
              paste("CLASP", subset, "| each measure alone without / with the language control"))
  print_table(summary_table %>% filter(subset == !!subset, model %in% c("full", "full + language")) %>% arrange(term, model),
              paste("CLASP", subset, "| full model without / with the language control"))
}
cat("finished\n")
