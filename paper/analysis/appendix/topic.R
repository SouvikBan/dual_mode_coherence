#!/usr/bin/env Rscript
# topic.R -- Appendix A, "Topic", and Section 5 (iii).
#
# (1) How often the learned-salience topic is the most-mentioned entity (and, when
# measures/topic_next_mention.py has been run, how often the topic of the next-mention model fitted
# on GUM is). (2) The topic bits alone: the subject bit s and the first-mention bit phi under the
# learned topic, under the most-mentioned entity and under the next-mention topic, with held-out
# gains on unseen text. (3) The subject account against the
# order-of-mention account on the versions / utterances whose subject bit and first-mention bit
# differ: each bit alone, both together, and the difference of the two coefficients in the joint
# model (Wald test from the fitted covariance).
#
# Output: results/appendix_topic_<dataset>.csv (bits alone), results/appendix_topic_<dataset>_subject_vs_first.csv
# Usage:  NCORES=22 Rscript topic.R <naturalstories|clasp>

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

dataset <- script_arguments()[1]
ds <- load_dataset(dataset)
EXTRA <- c("same_topic", "topic_is_subject", "topic_is_first", "count_topic_is_subject", "count_topic_is_first",
           "s_first", "count_s_subject", "count_s_first")
BITS <- c("s_subject", "s_first", "count_s_subject", "count_s_first")
has_next_mention_topic <- "nm_s_subject" %in% names(ds$measures)
if (has_next_mention_topic) {
  EXTRA <- c(EXTRA, "same_topic_as_most_mentioned", "nm_s_subject", "nm_s_first")
  BITS <- c(BITS, "nm_s_subject", "nm_s_first")
}
data_by_strategy <- setNames(lapply(STRATEGIES, function(strategy) {
  prepare_data(ds, strategy, extra_columns = EXTRA) %>%
    mutate(across(all_of(setdiff(BITS, "s_subject")), standardise, .names = "{.col}_s"))
}), STRATEGIES)

# ---- (1) topic definitions -----------------------------------------------------------------------
targets <- data_by_strategy[[DISPLAY_STRATEGY]] %>% distinct(across(all_of(measure_keys(dataset))), .keep_all = TRUE)
cat(sprintf("%s: %d targets with established entities; learned topic = most-mentioned entity in %.0f%%; subject bit and first bit differ on %d targets (learned topic) / %d (most-mentioned)\n",
            dataset, nrow(targets), 100 * mean(targets$same_topic), sum(targets$topic_is_subject != targets$topic_is_first),
            sum(targets$count_topic_is_subject != targets$count_topic_is_first)))
if (has_next_mention_topic) {
  cat(sprintf("%s: next-mention topic = most-mentioned entity in %.0f%% of the targets\n", dataset, 100 * mean(targets$same_topic_as_most_mentioned)))
}

# ---- (2) the bits alone -------------------------------------------------------------------------
spec <- function(label, terms, term, heldout = FALSE) list(label = label, terms = terms, term = term, heldout = heldout)
models <- c(list(control_model_spec()),
            lapply(BITS, function(bit) spec(paste(bit, "alone"), scaled(bit), scaled(bit), heldout = TRUE)),
            list(spec("s_subject | s_first", scaled(c("s_first", "s_subject")), "s_subject_s"),
                 spec("s_first | s_subject", scaled(c("s_first", "s_subject")), "s_first_s")))
results <- compare_models(ds, data_by_strategy, models)
summary_table <- summarise_comparison(results)
write_csv(results, file.path(RESULTS_DIR, paste0("appendix_topic_", dataset, "_per_strategy.csv")))
write_csv(summary_table, file.path(RESULTS_DIR, paste0("appendix_topic_", dataset, ".csv")))
print_table(summary_table, paste(dataset, "| topic bits | per SD at temperature 1.25 (stat); mean; strategies significant; AIC gain; held-out gain x100"))

# ---- (3) subject vs first mention where the two bits part ------------------------------------
compare_accounts <- function(data, subject_bit, first_bit, rows_label, topic_label, strategy) {
  safe_standardise <- function(x) if (sd(x) == 0) x * 0 else standardise(x)
  data$subject_s <- safe_standardise(data[[subject_bit]])
  data$first_s <- safe_standardise(data[[first_bit]])
  controls <- fit_model(model_formula(), data, dataset)
  subject_model <- fit_model(model_formula("subject_s"), data, dataset)
  first_model <- fit_model(model_formula("first_s"), data, dataset)
  joint_model <- fit_model(model_formula(c("subject_s", "first_s")), data, dataset)
  subject_coef <- fixed_effects(subject_model, dataset)
  first_coef <- fixed_effects(first_model, dataset)
  joint_coef <- fixed_effects(joint_model, dataset)
  covariance <- if (dataset == "naturalstories") as.matrix(vcov(joint_model)) else vcov(joint_model)$cond
  difference <- joint_coef["subject_s", 1] - joint_coef["first_s", 1]
  difference_se <- sqrt(covariance["subject_s", "subject_s"] + covariance["first_s", "first_s"] - 2 * covariance["subject_s", "first_s"])
  data.frame(
    topic = topic_label, rows = rows_label, strategy = strategy, n_observations = nrow(data),
    n_targets = n_distinct(data$sentence_key),
    subject_alone = subject_coef["subject_s", 1], subject_alone_stat = subject_coef["subject_s", 3],
    first_alone = first_coef["first_s", 1], first_alone_stat = first_coef["first_s", 3],
    subject_joint = joint_coef["subject_s", 1], subject_joint_stat = joint_coef["subject_s", 3],
    first_joint = joint_coef["first_s", 1], first_joint_stat = joint_coef["first_s", 3],
    difference = difference, difference_stat = difference / difference_se,
    aic_controls = AIC(controls), aic_subject = AIC(subject_model), aic_first = AIC(first_model), aic_joint = AIC(joint_model)
  )
}
TOPICS <- list(salience = c(subject = "s_subject", first = "s_first", subject_bit = "topic_is_subject", first_bit = "topic_is_first"),
               count = c(subject = "count_s_subject", first = "count_s_first", subject_bit = "count_topic_is_subject", first_bit = "count_topic_is_first"))
accounts <- bind_rows(mclapply(STRATEGIES, function(strategy) {
  data <- data_by_strategy[[strategy]]
  bind_rows(lapply(names(TOPICS), function(topic) {
    columns <- TOPICS[[topic]]
    disagreeing <- data %>% filter(.data[[columns["subject_bit"]]] != .data[[columns["first_bit"]]])
    bind_rows(compare_accounts(disagreeing, columns["subject"], columns["first"], "bits differ", topic, strategy),
              compare_accounts(data, columns["subject"], columns["first"], "all", topic, strategy))
  }))
}, mc.cores = N_CORES))
write_csv(accounts, file.path(RESULTS_DIR, paste0("appendix_topic_", dataset, "_subject_vs_first.csv")))
accounts_summary <- accounts %>%
  group_by(topic, rows) %>%
  summarise(n_targets = first(n_targets),
            subject_alone_t125 = first(subject_alone[strategy == DISPLAY_STRATEGY]), subject_alone_sig = sum(abs(subject_alone_stat) > 1.96),
            first_alone_t125 = first(first_alone[strategy == DISPLAY_STRATEGY]), first_alone_sig = sum(abs(first_alone_stat) > 1.96),
            difference_t125 = first(difference[strategy == DISPLAY_STRATEGY]), difference_sig = sum(abs(difference_stat) > 1.96),
            subject_better_aic = sum(aic_subject < aic_first), .groups = "drop") %>%
  mutate(across(where(is.numeric), ~ signif(.x, 3)))
print_table(accounts_summary, paste(dataset, "| subject bit vs first-mention bit (estimate per SD at temperature 1.25; strategies significant of 11)"))
cat("finished\n")
