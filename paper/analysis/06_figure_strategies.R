#!/usr/bin/env Rscript
# 06_figure_strategies.R -- Figure 3: estimates of the 15 measures in the full model under each of
# the 11 sampling strategies, with the mean over strategies and the number of strategies at
# p < .05 (bold with * marks p < .05). One grid per dataset: Natural Stories in ms per SD, CLASP
# in rating points per SD.
#
# Input:  results/estimates_naturalstories.csv, results/estimates_clasp_rating_points.csv,
#         results/clasp_rating_scale_factors.csv (sets the CLASP colour limit to 0.1 logit)
# Output: figures/fig_strategies_naturalstories.pdf/.png, figures/fig_strategies_clasp.pdf/.png
# Usage:  Rscript 06_figure_strategies.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))
suppressPackageStartupMessages(library(ggplot2))

STRATEGY_LABELS <- c(ancestral = "ancestral", temp_075 = "T = 0.75", temp_125 = "T = 1.25",
                     nucleus_08 = "p = 0.80", nucleus_085 = "p = 0.85", nucleus_09 = "p = 0.90", nucleus_095 = "p = 0.95",
                     typical_02 = "τ = 0.20", typical_03 = "τ = 0.30", typical_085 = "τ = 0.85", typical_095 = "τ = 0.95")
COLUMN_LEVELS <- c(STRATEGY_LABELS, "mean", "p < .05")
MEASURE_LABELS <- c(
  d_est = "bar(d)[est]", d_part = "bar(d)[part]", d_part_A = "bar(d)[part]^A", d_top = "bar(d)[top]", d_beta = "bar(d)[beta]",
  d_lex1 = "bar(d)[lex]^(1)", d_lex2 = "bar(d)[lex]^(2)", d_lex3 = "bar(d)[lex]^(3)",
  d_pos1 = "bar(d)[pos]^(1)", d_pos2 = "bar(d)[pos]^(2)", d_pos3 = "bar(d)[pos]^(3)",
  d_dep1 = "bar(d)[dep]^(1)", d_dep2 = "bar(d)[dep]^(2)", d_dep3 = "bar(d)[dep]^(3)", d_cos = "bar(d)[cos]"
)
N_ENTITY_ROWS <- length(ENTITY_MEASURES)

DATASETS <- list(
  naturalstories = list(file = "estimates_naturalstories.csv", unit = "ms per SD", digits = 0, limit = 200),
  clasp = list(file = "estimates_clasp_rating_points.csv", unit = "rating points per SD", digits = 3, limit = NA)
)
# CLASP colour limit: 0.1 logit expressed in rating points
factors <- read_csv(file.path(RESULTS_DIR, "clasp_rating_scale_factors.csv"), show_col_types = FALSE)
DATASETS$clasp$limit <- 0.1 * factors$factor_ame_cond[factors$model == "controls" & factors$strategy == DISPLAY_STRATEGY]

grid_cells <- function(estimates, digits) {
  estimates <- estimates %>%
    mutate(significant = abs(stat) > 1.96,
           measure = factor(term, levels = rev(FULL_MODEL_MEASURES)),
           column = STRATEGY_LABELS[strategy])
  per_measure <- estimates %>%
    group_by(measure) %>%
    summarise(mean = mean(estimate), n_significant = sum(significant), .groups = "drop")
  format_value <- function(value) format(round(value, digits), nsmall = digits, trim = TRUE)
  bind_rows(
    estimates %>% transmute(measure, column, value = estimate, significant,
                            text = paste0(format_value(estimate), ifelse(significant, "*", ""))),
    per_measure %>% transmute(measure, column = "mean", value = mean, significant = FALSE, text = format_value(mean)),
    per_measure %>% transmute(measure, column = "p < .05", value = NA_real_, significant = FALSE,
                              text = paste0(n_significant, "/", length(STRATEGIES)))
  ) %>%
    mutate(column = factor(column, levels = COLUMN_LEVELS))
}

grid_plot <- function(cells, title, limit) {
  ggplot(cells, aes(column, measure)) +
    geom_tile(aes(fill = value), colour = "white") +
    geom_text(aes(label = text, fontface = ifelse(significant, "bold", "plain")), size = 3) +
    scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
                         limits = c(-limit, limit), oob = scales::squish, na.value = "grey92", name = "estimate") +
    geom_vline(xintercept = length(STRATEGIES) + 0.5, colour = "grey40") +
    geom_hline(yintercept = length(FULL_MODEL_MEASURES) - N_ENTITY_ROWS + 0.5, colour = "grey40") +
    scale_x_discrete(position = "top") +
    scale_y_discrete(labels = function(terms) parse(text = MEASURE_LABELS[terms])) +
    labs(x = NULL, y = NULL, title = title) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 45, hjust = 0, vjust = 0), panel.grid = element_blank(),
          legend.position = "right", plot.title = element_text(size = 11))
}

for (dataset in names(DATASETS)) {
  spec <- DATASETS[[dataset]]
  estimates <- read_csv(file.path(RESULTS_DIR, spec$file), show_col_types = FALSE) %>%
    filter(subset == "all", model == "full", term %in% FULL_MODEL_MEASURES)
  title <- paste0(if (dataset == "naturalstories") "Natural Stories" else "CLASP", ", full model, ", spec$unit)
  plot <- grid_plot(grid_cells(estimates, spec$digits), title, spec$limit)
  ggsave(file.path(FIGURES_DIR, paste0("fig_strategies_", dataset, ".pdf")), plot, width = 13, height = 6.5, device = cairo_pdf)
  ggsave(file.path(FIGURES_DIR, paste0("fig_strategies_", dataset, ".png")), plot, width = 13, height = 6.5, dpi = 150)
  cat("saved", dataset, "\n")
}
