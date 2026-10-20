#!/usr/bin/env Rscript
# 04_heldout.R -- prediction of unseen text (paper, Appendix B, Tables 2 and 3).
#
# Models: controls; + 5 entity measures; + 10 non-entity measures; + all 15 (full); and the
# single measures named in the tables. Fold schemes: "text" = unseen stories (Natural Stories,
# one story per fold) or unseen contexts (CLASP, ten folds of ten contexts); "participant" =
# five folds by participant. All 11 strategies; CLASP on all contexts and on the 80 with
# established entities.
#
# Per fold: the held-out log-likelihood per observation (Gaussian with the predictive variance of
# an unseen story and utterance on Natural Stories; ordered beta with the context and version
# effects of an unseen context at 0 on CLASP), and R2 and RMSE on the response scale (ms;
# rating 1 to 4) from the fixed effects alone (r2, rmse) and from the fixed effects plus the
# participant intercept of the training fit (r2_participant, rmse_participant).
#
# Output: results/heldout_<dataset>_per_fold.csv
#         results/heldout_<dataset>.csv  mean and 95% t-interval over folds of the gain in
#                                        log-likelihood over the controls (x 100), after averaging
#                                        over strategies within fold; folds that beat the controls;
#                                        R2 and RMSE likewise
# Usage: NCORES=40 Rscript 04_heldout.R <naturalstories|clasp>

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

dataset <- script_arguments()[1]
stopifnot(dataset %in% c("naturalstories", "clasp"))
ds <- load_dataset(dataset)

MODELS <- list(
  controls = NULL,
  entity = scaled(ENTITY_MEASURES),
  nonentity = scaled(NON_ENTITY_MEASURES),
  full = scaled(FULL_MODEL_MEASURES)
)
if (dataset == "naturalstories") {
  MODELS <- c(MODELS, list(
    `d_est only` = "d_est_s",
    `beta only` = "beta_s",
    `d_est + beta` = c("d_est_s", "beta_s"),
    `d_cos only` = "d_cos_s"
  ))
} else {
  MODELS <- c(MODELS, list(
    `d_est only` = "d_est_s",
    `d_top only` = "d_top_s",
    `d_est + d_top` = c("d_est_s", "d_top_s"),
    `d_part_A only` = "d_part_A_s",
    `d_lex2 only` = "d_lex2_s"
  ))
}
SCHEMES <- c("text", "participant")

jobs <- expand.grid(subset = ds$subsets, strategy = STRATEGIES, scheme = SCHEMES, model = names(MODELS),
                    stringsAsFactors = FALSE)

evaluate_job <- function(j) {
  subset <- jobs$subset[j]
  strategy <- jobs$strategy[j]
  scheme <- jobs$scheme[j]
  model_name <- jobs$model[j]
  data <- prepare_data(ds, strategy, subset)
  formula <- model_formula(MODELS[[model_name]])
  folds <- if (scheme == "text") text_folds(ds, data) else participant_folds(data)
  rows <- list()
  for (k in sort(unique(folds))) {
    train <- data[folds != k, ]
    test <- data[folds == k, ]
    model <- tryCatch(fit_model(formula, train, dataset), error = function(e) NULL)
    if (is.null(model)) next
    prediction <- heldout_predictions(model, test, dataset, scheme)
    rows[[length(rows) + 1]] <- data.frame(
      subset = subset, strategy = strategy, scheme = scheme, model = model_name, fold = k, n = nrow(test),
      ll = prediction$ll,
      r2 = heldout_r2(test$y, prediction$mu_fixed, dataset),
      rmse = heldout_rmse(test$y, prediction$mu_fixed, dataset),
      r2_participant = heldout_r2(test$y, prediction$mu_participant, dataset),
      rmse_participant = heldout_rmse(test$y, prediction$mu_participant, dataset)
    )
  }
  bind_rows(rows)
}

per_fold <- bind_rows(mclapply(seq_len(nrow(jobs)), function(j) {
  tryCatch(evaluate_job(j), error = function(e) { cat("ERROR", j, conditionMessage(e), "\n", file = stderr()); NULL })
}, mc.cores = N_CORES))
write_csv(per_fold, file.path(RESULTS_DIR, paste0("heldout_", dataset, "_per_fold.csv")))

# ---- paired differences against the controls, folds as the unit -----------------------------
control_ll <- per_fold %>% filter(model == "controls") %>% select(subset, strategy, scheme, fold, ll_controls = ll)
by_fold <- per_fold %>%
  inner_join(control_ll, by = c("subset", "strategy", "scheme", "fold")) %>%
  mutate(dll = 100 * (ll - ll_controls)) %>%
  group_by(subset, scheme, model, fold) %>%
  summarise(across(c(dll, r2, rmse, r2_participant, rmse_participant), mean), .groups = "drop")

t_interval <- function(x) {
  half_width <- qt(0.975, length(x) - 1) * sd(x) / sqrt(length(x))
  c(mean = mean(x), lo = mean(x) - half_width, hi = mean(x) + half_width)
}
interval_columns <- function(x, prefix) {
  interval <- t_interval(x)
  setNames(as.list(interval), paste0(prefix, "_", names(interval)))
}

summary_table <- by_fold %>%
  group_by(subset, scheme, model) %>%
  summarise(
    n_folds = n(),
    folds_positive = sum(dll > 0),
    as_tibble(c(interval_columns(dll, "dll"), interval_columns(r2, "r2"), interval_columns(rmse, "rmse"),
                interval_columns(r2_participant, "r2_participant"), interval_columns(rmse_participant, "rmse_participant"))),
    .groups = "drop"
  )
write_csv(summary_table, file.path(RESULTS_DIR, paste0("heldout_", dataset, ".csv")))
print_table(summary_table %>% mutate(across(where(is.numeric), ~ signif(.x, 3))),
            paste(dataset, "| held-out gain over the controls (x 100 nats per observation) with 95% interval over folds"))
cat("finished\n")
