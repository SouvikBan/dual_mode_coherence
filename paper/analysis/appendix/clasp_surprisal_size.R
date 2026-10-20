#!/usr/bin/env Rscript
# clasp_surprisal_size.R -- Appendix A, "Surprisal model on CLASP".
#
# The control model of the paper (mean per-word surprisal + length + position + mentions in the
# context; ordered beta; participant, context and version intercepts) is fitted once per Pythia
# size on the same ratings; reported are the AIC, the AIC gain of surprisal over the other
# controls, and the surprisal coefficient per SD (logits and rating points).
#
# Input:  every pythia-*-full.tsv in CLASP_SURPRISAL_DIR (scripts/score_clasp.py per model size)
# Output: results/appendix_clasp_surprisal_size.csv
# Usage:  Rscript clasp_surprisal_size.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

factors <- read_csv(file.path(RESULTS_DIR, "clasp_rating_scale_factors.csv"), show_col_types = FALSE)
rating_points_per_logit <- factors$factor_ame_cond[factors$model == "controls" & factors$strategy == DISPLAY_STRATEGY]

files <- list.files(CLASP_SURPRISAL_DIR, pattern = "^pythia-.*-full\\.tsv$", full.names = TRUE)
rows <- list()
for (file in files) {
  model_name <- sub("-full\\.tsv$", "", basename(file))
  data <- load_clasp_responses(file) %>%
    mutate(across(all_of(CONTROL_VARIABLES), standardise, .names = "{.col}_s"))
  with_surprisal <- fit_model(model_formula(), data, "clasp")
  without_surprisal <- fit_model(model_formula(controls = "length_s + position_s + ctx_mentions_s"), data, "clasp")
  coefficients <- fixed_effects(with_surprisal, "clasp")
  rows[[length(rows) + 1]] <- data.frame(
    model = model_name, n = nrow(data), aic = AIC(with_surprisal),
    aic_gain_surprisal = AIC(without_surprisal) - AIC(with_surprisal),
    surprisal_logits_per_sd = coefficients["surprisal_s", 1],
    surprisal_rating_points_per_sd = coefficients["surprisal_s", 1] * rating_points_per_logit,
    z = coefficients["surprisal_s", 3])
  cat("done", model_name, "\n", file = stderr())
}
results <- bind_rows(rows) %>% arrange(aic)
write_csv(results, file.path(RESULTS_DIR, "appendix_clasp_surprisal_size.csv"))
print_table(results %>% mutate(across(where(is.numeric), ~ signif(.x, 5))), "CLASP control model by surprisal model")
cat("finished\n")
