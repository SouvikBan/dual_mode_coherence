#!/usr/bin/env Rscript
# 01_prepare_naturalstories.R -- one row per participant and utterance of Natural Stories.
#
# Words read in 200 to 2,000 ms by participants who answered at least four comprehension
# questions of the story correctly are kept (paper, Appendix A "Data"). Reading times and the
# word-level surprisal (Pythia-70m-deduped, step 1,000) are summed over the kept words of each
# utterance, so that the response, the surprisal control and the length control (words read)
# are computed on the same words.
#
# Input:  NS_READING_TIMES (processed_RTs.tsv), NS_WORD_SURPRISAL (word, item, zone, sentence_id, surprisal)
# Output: data/naturalstories_reading_times.csv
# Usage:  Rscript 01_prepare_naturalstories.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))

MIN_RT <- 200
MAX_RT <- 2000
MIN_CORRECT <- 4

reading_times <- read_tsv(NS_READING_TIMES, show_col_types = FALSE) %>% distinct()
cat("reading-time rows:", nrow(reading_times), "participants:", n_distinct(reading_times$WorkerId), "\n")

kept <- reading_times %>% filter(RT > MIN_RT, RT < MAX_RT, correct >= MIN_CORRECT)
cat(sprintf("rows after the word-level filter: %d (%.1f%% kept)\n", nrow(kept), 100 * nrow(kept) / nrow(reading_times)))

surprisal <- read_tsv(NS_WORD_SURPRISAL, show_col_types = FALSE)
stopifnot(identical(names(surprisal), c("word", "item", "zone", "sentence_id", "surprisal")))
stopifnot(!any(duplicated(surprisal[c("item", "zone")])))

per_word <- kept %>% inner_join(surprisal %>% select(item, zone, sentence_id, surprisal), by = c("item", "zone"))
cat("rows matched to a surprisal value:", nrow(per_word), "of", nrow(kept), "\n")

per_utterance <- per_word %>%
  group_by(item, sentence_id, WorkerId) %>%
  summarise(sum_RT = sum(RT),
            mean_RT = mean(RT),
            n_words_read = n(),
            sum_surprisal = sum(surprisal),
            mean_surprisal = mean(surprisal),
            .groups = "drop") %>%
  left_join(surprisal %>% count(item, sentence_id, name = "n_words_sentence"), by = c("item", "sentence_id"))

write_csv(per_utterance, NS_SENTENCE_TABLE)
cat("wrote", NS_SENTENCE_TABLE, ":", nrow(per_utterance), "rows,",
    n_distinct(per_utterance$WorkerId), "participants,",
    n_distinct(paste(per_utterance$item, per_utterance$sentence_id)), "utterances\n")
