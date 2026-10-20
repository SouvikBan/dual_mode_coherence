# setup.R -- shared code of the paper's analyses: constants, data loading, model fitting and
# held-out evaluation. Every analysis script sources this file after setting PAPER_DIR.
#
# Models (paper, Section 4 "Statistical modeling"):
#   Natural Stories  summed reading time of an utterance per participant, linear mixed model,
#                    random intercepts for participant, story and utterance;
#   CLASP            rating per participant and version on (rating - 1) / 3, ordered beta
#                    regression, random intercepts for participant, context and version.
#   Controls         surprisal, length, position in the text, mentions in the context; every
#                    predictor standardised. Natural Stories: summed surprisal (Pythia-70m,
#                    step 1,000) and length in words read; CLASP: mean per-word surprisal
#                    (Pythia-12b) and length of context plus sentence in words.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(lme4)
  library(glmmTMB)
  library(parallel)
})

REPO_DIR <- normalizePath(file.path(PAPER_DIR, ".."))
source(file.path(PAPER_DIR, "analysis", "paths.R"))

options(width = 250)
N_CORES <- as.integer(Sys.getenv("NCORES", "11"))

# ---- constants ---------------------------------------------------------------------------------

STRATEGIES <- c("ancestral", "temp_075", "temp_125", "nucleus_08", "nucleus_085", "nucleus_09",
                "nucleus_095", "typical_02", "typical_03", "typical_085", "typical_095")
DISPLAY_STRATEGY <- "temp_125"   # Figure 1 and Table 1 show temperature 1.25

# The five information values of the entity model (paper, Section 4 "Statistical modeling").
ENTITY_MEASURES <- c("d_est", "d_part", "d_part_A", "d_top", "d_beta")
# The ten non-entity measures.
NON_ENTITY_MEASURES <- c("d_lex1", "d_lex2", "d_lex3", "d_pos1", "d_pos2", "d_pos3",
                         "d_dep1", "d_dep2", "d_dep3", "d_cos")
# Added alone to the controls only: the linking cost, its anticipatory form, the topic-is-subject
# bit, and the number of new entities (Appendix A).
ALONE_ONLY_MEASURES <- c("beta", "d_beta_A", "s_subject", "n_new")

FULL_MODEL_MEASURES <- c(ENTITY_MEASURES, NON_ENTITY_MEASURES)
ALL_MEASURES <- c(FULL_MODEL_MEASURES, ALONE_ONLY_MEASURES)

CONTROL_VARIABLES <- c("surprisal", "length", "position", "ctx_mentions")
CONTROLS <- paste0(CONTROL_VARIABLES, "_s", collapse = " + ")
RANDOM_EFFECTS <- "(1 | WorkerId) + (1 | group) + (1 | sentence_key)"

N_TEXT_FOLDS_CLASP <- 10      # folds of ten contexts
N_PARTICIPANT_FOLDS <- 5
CLASP_LANGUAGES <- c("0" = "English", "1" = "Czech", "2" = "German", "3" = "Spanish", "4" = "French")

standardise <- function(x) as.vector(scale(x))

scaled <- function(variables) paste0(variables, "_s")

significance_stars <- function(stat) {
  p <- 2 * pnorm(-abs(stat))
  ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", "")))
}

# ---- measures --------------------------------------------------------------------------------

read_measure_files <- function(dataset, measure) {
  files <- list.files(file.path(MEASURES_DIR, measure, dataset), pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0) stop("no measure files under ", file.path(MEASURES_DIR, measure, dataset))
  bind_rows(lapply(files, read_csv, show_col_types = FALSE, guess_max = 10000))
}

# One row per target and strategy with the paper's measure names.
load_measures <- function(dataset) {
  keys <- c("item_id", "strategy")
  established <- read_measure_files(dataset, "established_entities") %>%
    select(item_id, document_id, sentence_index, strategy, n_established, n_mentioned_established, n_new,
           d_est, d_est_A = d_est_anticipatory)
  part <- read_measure_files(dataset, "part_of_story") %>%
    select(item_id, strategy, d_part, d_part_A = d_part_anticipatory,
           d_part_history, d_part_history_A = d_part_history_anticipatory)
  linking <- read_measure_files(dataset, "linking_cost") %>%
    select(item_id, strategy, beta, d_beta, d_beta_A = d_beta_anticipatory)
  topic <- read_measure_files(dataset, "topic") %>%
    select(item_id, strategy, same_topic, topic_is_subject, topic_is_first, count_topic_is_subject, count_topic_is_first,
           s_subject, s_first, count_s_subject, count_s_first, d_top, d_top_A = d_top_anticipatory)
  non_entity <- read_measure_files(dataset, "non_entity") %>%
    select(item_id, strategy, all_of(NON_ENTITY_MEASURES))
  measures <- established %>%
    inner_join(part, by = keys) %>%
    inner_join(linking, by = keys) %>%
    inner_join(topic, by = keys) %>%
    inner_join(non_entity, by = keys)
  if (dataset == "naturalstories") {
    measures <- measures %>%
      mutate(item = as.integer(str_split_i(document_id, "_", 2)), sentence_id = sentence_index + 1L)
  }
  # entity measures are 0 by construction when the context has no established entity
  entity_columns <- c(ENTITY_MEASURES, "d_est_A", "d_part_history", "d_part_history_A", "d_top_A",
                      "beta", "d_beta_A", "s_subject", "s_first", "count_s_subject", "count_s_first")
  measures <- measures %>% mutate(across(all_of(entity_columns), ~ ifelse(is.na(.x), 0, .x)))
  # the measures of Appendix A "Claims tested and not reported above" and the next-mention topic,
  # when they have been computed (NA otherwise)
  measures %>%
    join_optional_measure(dataset, "referring_forms",
                          c("d_form_accessibility", "d_form_givenness", "d_form_first_accessibility", "d_form_first_givenness")) %>%
    join_optional_measure(dataset, "topic_next_mention",
                          c("same_topic_as_most_mentioned", "nm_topic_is_subject", "nm_topic_is_first", "nm_s_subject", "nm_s_first")) %>%
    join_optional_measure(dataset, "next_mention",
                          c("n_previous_entities", "log2_M", "first_in_previous", "first_kind",
                            paste0("surprisal_first_", NEXT_MENTION_ACCOUNTS), paste0("entropy_first_", NEXT_MENTION_ACCOUNTS),
                            "pronoun_eligible", "n_candidates", "surprisal_uniform", "surprisal_expectancy",
                            "surprisal_mirror", "surprisal_bayesian"))
}

NEXT_MENTION_ACCOUNTS <- c("grammatical", "recency", "frequency", "infostatus", "full")

join_optional_measure <- function(measures, dataset, measure, columns) {
  directory <- file.path(MEASURES_DIR, measure, dataset)
  if (!dir.exists(directory) || length(list.files(directory, pattern = "\\.csv$")) == 0) return(measures)
  extra <- read_measure_files(dataset, measure)
  extra <- extra %>% select(item_id, strategy, any_of(columns))
  measures %>% left_join(extra, by = c("item_id", "strategy"))
}

measure_keys <- function(dataset) if (dataset == "naturalstories") c("item", "sentence_id") else "item_id"

# ---- responses and controls -----------------------------------------------------------------

# Natural Stories: one row per participant and utterance (from 01_prepare_naturalstories.R),
# with the number of gold mentions in the context before the utterance as a control.
load_naturalstories_responses <- function() {
  context_mentions <- read_csv(NS_GOLD_COUNTS, show_col_types = FALSE) %>%
    mutate(sentence_id = sent_id + 1L) %>%
    select(item, sentence_id, ctx_mentions = ctx_n_mentions)
  read_csv(NS_SENTENCE_TABLE, show_col_types = FALSE) %>%
    transmute(item, sentence_id, WorkerId,
              y = sum_RT, surprisal = sum_surprisal, length = n_words_read, position = sentence_id,
              group = item, sentence_key = paste(item, sentence_id, sep = ":")) %>%
    inner_join(context_mentions, by = c("item", "sentence_id"))
}

# CLASP: one row per in-context rating of a version, from the scorer's output.
read_clasp_ratings <- function(surprisal_file) {
  read_tsv(surprisal_file, show_col_types = FALSE) %>%
    transmute(WorkerId,
              item_id = paste0("clasp_", sent_id, "::", CLASP_LANGUAGES[as.character(lang_id)]),
              group = sent_id,
              rating = suppressWarnings(as.numeric(answer)),
              sentence_words = str_count(sent, " ") + 1,
              surprisal = sentence_surprisal / sentence_words,
              length = ifelse(is.na(prefix), 1, str_count(prefix, " ") + 1) + sentence_words) %>%
    filter(!is.na(rating)) %>%
    distinct(WorkerId, item_id, .keep_all = TRUE) %>%
    mutate(y = (rating - 1) / 3, sentence_key = item_id) %>%
    select(-sentence_words)
}

load_clasp_responses <- function(surprisal_file = CLASP_RATINGS) {
  positions <- read_csv(CLASP_POSITIONS, show_col_types = FALSE) %>%
    transmute(item_id, position = sentence_index)
  context_mentions <- read_csv(CLASP_GOLD_COUNTS, show_col_types = FALSE) %>%
    select(item_id, ctx_mentions = ctx_n_mentions)
  read_clasp_ratings(surprisal_file) %>%
    inner_join(positions, by = "item_id") %>%
    inner_join(context_mentions, by = "item_id")
}

load_responses <- function(dataset) {
  if (dataset == "naturalstories") load_naturalstories_responses() else load_clasp_responses()
}

# Everything an analysis needs for one dataset: responses, measures, fold assignments.
load_dataset <- function(dataset) {
  responses <- load_responses(dataset)
  list(
    name = dataset,
    responses = responses,
    measures = load_measures(dataset),
    subsets = if (dataset == "naturalstories") "all" else c("all", "nonempty"),
    context_folds = if (dataset == "clasp") clasp_context_folds(responses$group) else NULL
  )
}

# Join responses and the measures of one strategy; drop story-initial utterances (no established
# entities) on Natural Stories, or restrict CLASP to the 80 contexts with established entities;
# standardise every predictor.
prepare_data <- function(ds, strategy, subset = "all", extra_columns = character(0)) {
  keys <- measure_keys(ds$name)
  measures <- ds$measures %>%
    filter(strategy == !!strategy) %>%
    select(all_of(keys), n_established, n_mentioned_established, all_of(ALL_MEASURES), all_of(extra_columns))
  data <- ds$responses %>% inner_join(measures, by = keys)
  if (ds$name == "naturalstories" || subset == "nonempty") data <- data %>% filter(n_established > 0)
  data %>%
    filter(if_all(all_of(ALL_MEASURES), ~ !is.na(.x))) %>%
    mutate(across(all_of(c(CONTROL_VARIABLES, ALL_MEASURES)), standardise, .names = "{.col}_s"))
}

# ---- models -----------------------------------------------------------------------------------

model_formula <- function(fixed_terms = NULL, controls = CONTROLS) {
  right_hand_side <- paste(c(controls, fixed_terms, RANDOM_EFFECTS), collapse = " + ")
  as.formula(paste("y ~", right_hand_side))
}

fit_model <- function(formula, data, dataset) {
  if (dataset == "naturalstories") {
    lmer(formula, data = data, REML = FALSE, control = lmerControl(calc.derivs = FALSE))
  } else {
    glmmTMB(formula, data = data, family = ordbeta())
  }
}

fixed_effects <- function(model, dataset) {
  if (dataset == "naturalstories") summary(model)$coefficients else summary(model)$coefficients$cond
}

coefficient_row <- function(model, dataset, term) {
  table <- fixed_effects(model, dataset)
  data.frame(estimate = table[term, 1], se = table[term, 2], stat = table[term, 3])
}

# ---- ordered beta helpers (CLASP) -----------------------------------------------------------

ordbeta_cutpoints <- function(model) model$fit$par[grep("psi", names(model$fit$par))]

# log density of the ordered beta with point masses at 0 and 1 (glmmTMB parametrisation)
ordbeta_log_density <- function(y, mu, phi, cutpoints) {
  eta <- qlogis(mu)
  p_zero <- 1 - plogis(eta - cutpoints[1])
  p_one <- plogis(eta - cutpoints[2])
  p_middle <- pmax(plogis(eta - cutpoints[1]) - plogis(eta - cutpoints[2]), 1e-12)
  y_inner <- pmin(pmax(y, 1e-6), 1 - 1e-6)
  ifelse(y <= 0, log(pmax(p_zero, 1e-12)),
         ifelse(y >= 1, log(pmax(p_one, 1e-12)),
                log(p_middle) + dbeta(y_inner, mu * phi, (1 - mu) * phi, log = TRUE)))
}

# expected rating on the 0-1 scale: P(y = 1) + P(0 < y < 1) * plogis(eta)
ordbeta_expected <- function(eta, cutpoints) {
  p_zero <- 1 - plogis(eta - cutpoints[1])
  p_one <- plogis(eta - cutpoints[2])
  p_one + (1 - p_zero - p_one) * plogis(eta)
}

# derivative of the expected rating with respect to eta (for the rating-scale conversion)
ordbeta_expected_slope <- function(eta, cutpoints) {
  logistic_slope <- function(x) plogis(x) * (1 - plogis(x))
  p_zero <- 1 - plogis(eta - cutpoints[1])
  p_one <- plogis(eta - cutpoints[2])
  p_middle <- 1 - p_zero - p_one
  d_zero <- -logistic_slope(eta - cutpoints[1])
  d_one <- logistic_slope(eta - cutpoints[2])
  d_middle <- -d_zero - d_one
  d_one + d_middle * plogis(eta) + p_middle * logistic_slope(eta)
}

# ---- held-out evaluation ------------------------------------------------------------------

# Natural Stories: stories are the folds (leave one story out).
# CLASP: ten folds of ten contexts, assigned once with a fixed seed.
clasp_context_folds <- function(contexts) {
  contexts <- sort(unique(contexts))
  set.seed(1)
  setNames(sample(rep(seq_len(N_TEXT_FOLDS_CLASP), length.out = length(contexts))), contexts)
}

text_folds <- function(ds, data) {
  if (ds$name == "naturalstories") data$group else ds$context_folds[as.character(data$group)]
}

participant_folds <- function(data) {
  participants <- sort(unique(data$WorkerId))
  set.seed(2)
  assignment <- setNames(sample(rep(seq_len(N_PARTICIPANT_FOLDS), length.out = length(participants))), participants)
  assignment[as.character(data$WorkerId)]
}

# Predictions for a held-out set. Returns the mean log-likelihood per observation, the prediction
# from the fixed effects alone, and the prediction from the fixed effects plus the participant
# intercept of the training fit (participants recur across text folds; an unseen participant,
# story, utterance, context or version gets 0).
#   scheme "text":        unseen stories / contexts; their intercept variances enter the
#                         predictive variance on Natural Stories (sigma^2 + tau^2_story + tau^2_utterance)
#   scheme "participant": unseen participants; their intercept variance enters instead
heldout_predictions <- function(model, test, dataset, scheme = "text") {
  if (dataset == "naturalstories") {
    unseen_groups <- if (scheme == "text") c("group", "sentence_key") else "WorkerId"
    variances <- as.data.frame(VarCorr(model))
    predictive_variance <- sigma(model)^2 +
      sum(variances$vcov[variances$grp %in% unseen_groups & is.na(variances$var2)])
    mu <- predict(model, newdata = test, allow.new.levels = TRUE)
    mu_fixed <- predict(model, newdata = test, re.form = NA)
    mu_participant <- if (scheme == "text") {
      predict(model, newdata = test, re.form = ~ (1 | WorkerId), allow.new.levels = TRUE)
    } else {
      mu_fixed
    }
    list(ll = mean(dnorm(test$y, mu, sqrt(predictive_variance), log = TRUE)),
         mu_fixed = mu_fixed, mu_participant = mu_participant)
  } else {
    cutpoints <- ordbeta_cutpoints(model)
    mu <- predict(model, newdata = test, type = "response", allow.new.levels = TRUE)
    eta_fixed <- predict(model, newdata = test, type = "link", re.form = NA)
    participant_effect <- ranef(model)$cond$WorkerId[as.character(test$WorkerId), 1]
    participant_effect[is.na(participant_effect)] <- 0
    list(ll = mean(ordbeta_log_density(test$y, mu, sigma(model), cutpoints)),
         mu_fixed = ordbeta_expected(eta_fixed, cutpoints),
         mu_participant = ordbeta_expected(eta_fixed + participant_effect, cutpoints))
  }
}

# R2 and RMSE on the response scale of the paper: ms on Natural Stories, the 1-4 rating on CLASP.
response_scale <- function(values, dataset) if (dataset == "naturalstories") values else 1 + 3 * values

heldout_r2 <- function(y, mu, dataset) {
  y <- response_scale(y, dataset)
  mu <- response_scale(mu, dataset)
  1 - sum((y - mu)^2) / sum((y - mean(y))^2)
}

heldout_rmse <- function(y, mu, dataset) {
  sqrt(mean((response_scale(y, dataset) - response_scale(mu, dataset))^2))
}

# Mean held-out log-likelihood per observation of one formula over the text folds; used by the
# appendix scripts. Returns one value per fold.
fold_log_likelihoods <- function(ds, formula, data) {
  folds <- text_folds(ds, data)
  sapply(sort(unique(folds)), function(k) {
    model <- fit_model(formula, data[folds != k, ], ds$name)
    heldout_predictions(model, data[folds == k, ], ds$name)$ll
  })
}

# ---- generic model comparison for the appendix ------------------------------------------

# Fit a set of models on every strategy (in parallel) and report each model's term of interest
# against the control model: estimate per SD and statistic at the display strategy, mean estimate,
# strategies significant, AIC gain, and (when asked) the held-out gain over the controls.
#   data_by_strategy: named list of prepared data frames, one per strategy
#   models: list of list(label, terms (fixed-effect string or NULL), term (coefficient to report
#           or NULL), heldout (logical))
compare_models <- function(ds, data_by_strategy, models, controls = CONTROLS) {
  jobs <- expand.grid(strategy = names(data_by_strategy), model = seq_along(models), stringsAsFactors = FALSE)
  one_job <- function(j) {
    strategy <- jobs$strategy[j]
    spec <- models[[jobs$model[j]]]
    data <- data_by_strategy[[strategy]]
    if (!is.null(spec$rows)) data <- spec$rows(data)
    formula <- model_formula(spec$terms, controls)
    model <- fit_model(formula, data, ds$name)
    row <- data.frame(model = spec$label, strategy = strategy, n = nrow(data), aic = AIC(model),
                      estimate = NA_real_, stat = NA_real_)
    if (!is.null(spec$term)) {
      coefficients <- fixed_effects(model, ds$name)
      row$estimate <- coefficients[spec$term, 1]
      row$stat <- coefficients[spec$term, 3]
    }
    if (isTRUE(spec$heldout)) {
      row$fold_ll <- I(list(fold_log_likelihoods(ds, formula, data)))
    } else {
      row$fold_ll <- I(list(NULL))
    }
    row
  }
  results <- bind_rows(mclapply(seq_len(nrow(jobs)), function(j) {
    tryCatch(one_job(j), error = function(e) {
      cat("ERROR", models[[jobs$model[j]]]$label, jobs$strategy[j], conditionMessage(e), "\n", file = stderr())
      NULL
    })
  }, mc.cores = N_CORES))
  control_label <- models[[1]]$label
  results %>%
    group_by(strategy) %>%
    mutate(aic_gain = aic[model == control_label] - aic,
           dll_x100 = sapply(fold_ll, function(ll) {
             control_ll <- fold_ll[model == control_label][[1]]
             if (is.null(ll) || is.null(control_ll)) NA_real_ else 100 * mean(ll - control_ll)
           }),
           folds_positive = sapply(fold_ll, function(ll) {
             control_ll <- fold_ll[model == control_label][[1]]
             if (is.null(ll) || is.null(control_ll)) NA_real_ else sum(ll > control_ll)
           })) %>%
    ungroup() %>%
    filter(model != control_label) %>%
    select(-fold_ll)
}

control_model_spec <- function() list(label = "controls", terms = NULL, term = NULL, heldout = TRUE)

# Summarise compare_models() output over strategies.
summarise_comparison <- function(results, by = "model") {
  results %>%
    group_by(across(all_of(by))) %>%
    summarise(
      n = first(n),
      estimate_t125 = signif(first(estimate[strategy == DISPLAY_STRATEGY]), 3),
      stat_t125 = round(first(stat[strategy == DISPLAY_STRATEGY]), 1),
      estimate_mean = signif(mean(estimate), 3),
      n_significant = sum(abs(stat) > 1.96),
      n_strategies = n(),
      aic_gain_t125 = round(first(aic_gain[strategy == DISPLAY_STRATEGY]), 1),
      aic_gain_mean = round(mean(aic_gain), 1),
      dll_x100 = round(mean(dll_x100), 2),
      folds_positive = round(mean(folds_positive), 1),
      .groups = "drop"
    )
}

print_table <- function(table, title) {
  cat("\n######## ", title, " ########\n", sep = "")
  print(as.data.frame(table), row.names = FALSE)
}

# ---- script location -------------------------------------------------------------------------

script_arguments <- function() commandArgs(trailingOnly = TRUE)
