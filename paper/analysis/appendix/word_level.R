#!/usr/bin/env Rscript
# word_level.R -- Appendix A, "Word-level localization" (Natural Stories, temperature 1.25).
#
# Word reading times (200 to 2,000 ms, comprehension >= 4) are modelled with the word's own
# surprisal (Pythia-70m at step 1,000, recomputed per word in bits), the previous word's surprisal,
# length, previous length, Zipf frequency and position in the story, with participant, story and
# word intercepts. The utterance-level measure is added separately in five regions of the
# utterance: the word before the head of the target's first mention, the head itself, each of the
# two following words, and the remaining words.
#
# Input:  data/naturalstories_first_mention_zones.csv, data/naturalstories_word_surprisal.csv,
#         data/naturalstories_word_zipf.csv (measures/run_measures.sh), the reading times, the
#         linking-cost and established-entity measures
# Output: results/appendix_word_level.csv
# Usage:  Rscript word_level.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "../.."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

MIN_ROWS <- 300
REGIONS <- c("pre", "mention", "spill1", "spill2", "rest")

# ---- word table: surprisal, frequency, length, position, sentence of every zone -----------------
word_reading_times <- read_tsv(NS_READING_TIMES, show_col_types = FALSE) %>%
  distinct() %>%
  filter(RT > 200, RT < 2000, correct >= 4) %>%
  select(WorkerId, item, zone, RT)
sentence_of_zone <- read_tsv(NS_WORD_SURPRISAL, show_col_types = FALSE) %>%
  distinct(item, zone, .keep_all = TRUE) %>%
  select(item, zone, sentence_id)
words <- read_csv(NS_WORD_SURPRISAL_RECOMPUTED, show_col_types = FALSE) %>%
  select(item, zone, word, surprisal = surprisal_bits) %>%
  inner_join(read_csv(NS_WORD_ZIPF, show_col_types = FALSE) %>% select(item, zone, zipf), by = c("item", "zone")) %>%
  inner_join(sentence_of_zone, by = c("item", "zone")) %>%
  arrange(item, zone) %>%
  group_by(item) %>%
  mutate(length = nchar(word), previous_surprisal = lag(surprisal), previous_length = lag(length), story_position = zone / max(zone)) %>%
  ungroup()

# ---- utterance measures at temperature 1.25 and the zones of the first mention -----------------
measures <- load_measures("naturalstories") %>%
  filter(strategy == DISPLAY_STRATEGY) %>%
  select(item, sentence_id, beta, n_new)
first_mention <- read_csv(NS_FIRST_MENTION_ZONES, show_col_types = FALSE) %>%
  mutate(item = as.integer(str_split_i(document_id, "_", 2)), sentence_id = sentence_index + 1L) %>%
  filter(!is.na(first_zone_head)) %>%
  select(item, sentence_id, first_zone_head, first_zone_end)
utterances <- measures %>% inner_join(first_mention, by = c("item", "sentence_id"))

# ---- region rows ------------------------------------------------------------------------------
near_mention <- bind_rows(
  utterances %>% transmute(item, sentence_id, region = "pre", zone = first_zone_head - 1L),
  utterances %>% transmute(item, sentence_id, region = "mention", zone = first_zone_head),
  utterances %>% transmute(item, sentence_id, region = "spill1", zone = first_zone_end + 1L),
  utterances %>% transmute(item, sentence_id, region = "spill2", zone = first_zone_end + 2L)
) %>%
  inner_join(words %>% select(item, zone, zone_sentence = sentence_id), by = c("item", "zone")) %>%
  filter(zone_sentence == sentence_id | region == "pre") %>%     # spillover words must stay inside the utterance
  select(-zone_sentence)
remaining <- words %>%
  select(item, zone, sentence_id) %>%
  inner_join(utterances %>% select(item, sentence_id, first_zone_head, first_zone_end), by = c("item", "sentence_id")) %>%
  filter(zone < first_zone_head - 1 | zone > first_zone_end + 2) %>%
  transmute(item, sentence_id, region = "rest", zone)
regions <- bind_rows(near_mention, remaining)

data <- regions %>%
  inner_join(words %>% select(-sentence_id), by = c("item", "zone")) %>%
  inner_join(word_reading_times, by = c("item", "zone")) %>%
  inner_join(utterances, by = c("item", "sentence_id")) %>%
  filter(!is.na(previous_surprisal)) %>%
  mutate(zone_key = paste(item, zone),
         across(c(surprisal, previous_surprisal, length, previous_length, zipf, story_position), standardise, .names = "{.col}_s"))
cat("rows per region:\n")
print(table(data$region))

WORD_CONTROLS <- "surprisal_s + previous_surprisal_s + length_s + previous_length_s + zipf_s + story_position_s"
WORD_RANDOM_EFFECTS <- "(1 | WorkerId) + (1 | item) + (1 | zone_key)"
word_formula <- function(extra = NULL) as.formula(paste("RT ~", paste(c(WORD_CONTROLS, extra, WORD_RANDOM_EFFECTS), collapse = " + ")))
fit_word_model <- function(formula, data) lmer(formula, data = data, REML = FALSE, control = lmerControl(calc.derivs = FALSE))

MEASURE_SPECS <- list(
  list(label = "linking cost beta", variable = "beta", given = NULL),
  list(label = "new entities", variable = "n_new", given = NULL),
  list(label = "linking cost beta | new entities", variable = "beta", given = "n_new")
)
rows <- list()
for (spec in MEASURE_SPECS) {
  for (region in REGIONS) {
    subset <- data %>% filter(region == !!region)
    if (nrow(subset) < MIN_ROWS) next
    subset$x_s <- standardise(subset[[spec$variable]])
    given <- NULL
    if (!is.null(spec$given)) {
      subset$given_s <- standardise(subset[[spec$given]])
      given <- "given_s"
    }
    baseline <- fit_word_model(word_formula(given), subset)
    model <- fit_word_model(word_formula(c(given, "x_s")), subset)
    coefficients <- summary(model)$coefficients
    rows[[length(rows) + 1]] <- data.frame(
      measure = spec$label, region = region, n_rows = nrow(subset), n_words = n_distinct(subset$zone_key),
      ms_per_sd = round(coefficients["x_s", 1], 1), t = round(coefficients["x_s", 3], 1),
      aic_gain = round(AIC(baseline) - AIC(model), 1))
  }
}
results <- bind_rows(rows)
write_csv(results, file.path(RESULTS_DIR, "appendix_word_level.csv"))
print_table(results %>% select(measure, region, ms_per_sd, t, aic_gain, n_rows) %>%
              pivot_wider(names_from = region, values_from = c(ms_per_sd, t, aic_gain, n_rows)),
            "word reading time by region | ms per SD of the utterance measure (t): pre = word before the first mention, mention = its head, spill1/2 = the next two words, rest = other words")
cat("finished\n")
