#!/usr/bin/env Rscript
# 05_figure_estimates.R -- Figure 1 (effects of each measure added alone to the controls at
# temperature 1.25) and Figure 2 (the same with the controls, estimated in the full model).
#
# Top panel: Natural Stories, ms per SD; bottom panel: CLASP, rating points per SD. Error bars are
# 95% confidence intervals, stars mark p < .05 / .01 / .001 from the Wald statistic, and the
# outlined bar is the largest information value that is significant under all 11 strategies.
# In Figure 2 the Natural Stories length bar is cut at the axis limit and labelled with its value.
#
# Input:  results/estimates_naturalstories.csv, results/estimates_clasp_rating_points.csv
# Output: figures/fig_estimates.pdf/.png, figures/fig_estimates_controls.pdf/.png,
#         results/figure_estimates_data.csv
# Usage:  Rscript 05_figure_estimates.R

PAPER_DIR <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(PAPER_DIR, "analysis", "setup.R"))
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })

# plotting order, plotmath labels and groups of the predictors
PREDICTORS <- c("surprisal", "length", "position", "ctx_mentions",
                "d_est", "d_part", "d_part_A", "d_top", "s_subject", "beta", "d_beta", "d_beta_A",
                "d_lex1", "d_lex2", "d_lex3", "d_pos1", "d_pos2", "d_pos3", "d_dep1", "d_dep2", "d_dep3", "d_cos")
LABELS <- expression(
  "Surprisal", "Length", "Position", "Mentions",
  bar(d)[est], bar(d)[part], bar(d)[part]^A, bar(d)[top], s[U[n]], beta(U[n]), bar(d)[beta], bar(d)[beta]^A,
  bar(d)[lex]^(1), bar(d)[lex]^(2), bar(d)[lex]^(3), bar(d)[pos]^(1), bar(d)[pos]^(2), bar(d)[pos]^(3),
  bar(d)[dep]^(1), bar(d)[dep]^(2), bar(d)[dep]^(3), bar(d)[cos]
)
GROUPS <- c(rep("Controls", 4), "(i)", "(ii)", "(ii)", "(iii)", "(iii)", "(iv)", "(iv)", "(iv)",
            rep("Lexical", 3), rep("POS", 3), rep("Syntactic", 3), "Semantic")
names(GROUPS) <- PREDICTORS
GROUP_LEVELS <- c("Controls", "(i)", "(ii)", "(iii)", "(iv)", "Lexical", "POS", "Syntactic", "Semantic")
GROUP_COLOURS <- c(Controls = "grey55", `(i)` = "#D55E00", `(ii)` = "#E69F00", `(iii)` = "#CC79A7", `(iv)` = "#882255",
                   Lexical = "#56B4E9", POS = "#0072B2", Syntactic = "#009E73", Semantic = "#999933")
# information values that can carry the outline (not the anticipatory forms, the bit or the cost)
OUTLINE_CANDIDATES <- c("d_est", "d_part", "d_top", "d_beta", NON_ENTITY_MEASURES)
LENGTH_CAP_MS <- 650   # Figure 2: the Natural Stories length bar is cut here

# estimates of one dataset: every measure alone (temperature 1.25) with the number of strategies at
# p < .05; the controls from the full model
load_estimates <- function(file, dataset_label) {
  estimates <- read_csv(file, show_col_types = FALSE) %>% filter(subset == "all")
  strategies_significant <- estimates %>%
    filter(model == "alone") %>%
    group_by(term) %>%
    summarise(n_significant = sum(abs(stat) > 1.96), .groups = "drop")
  alone <- estimates %>%
    filter(model == "alone", strategy == DISPLAY_STRATEGY) %>%
    select(term, estimate, se, stat) %>%
    inner_join(strategies_significant, by = "term")
  controls <- estimates %>%
    filter(model == "full", strategy == DISPLAY_STRATEGY, term %in% CONTROL_VARIABLES) %>%
    select(term, estimate, se, stat) %>%
    mutate(n_significant = length(STRATEGIES))
  bind_rows(controls, alone) %>%
    filter(term %in% PREDICTORS) %>%
    mutate(dataset = dataset_label,
           stars = significance_stars(stat),
           group = factor(GROUPS[term], levels = GROUP_LEVELS),
           term = factor(term, levels = PREDICTORS))
}

estimates <- bind_rows(
  load_estimates(file.path(RESULTS_DIR, "estimates_naturalstories.csv"), "Natural Stories"),
  load_estimates(file.path(RESULTS_DIR, "estimates_clasp_rating_points.csv"), "CLASP")
)

# bars, intervals, clipping and the outline for one panel
panel_data <- function(estimates, dataset_label, with_controls) {
  data <- estimates %>% filter(dataset == dataset_label)
  if (!with_controls) data <- data %>% filter(group != "Controls")
  cap <- if (with_controls && dataset_label == "Natural Stories") LENGTH_CAP_MS else Inf
  data %>%
    mutate(
      clipped = abs(estimate) > cap,
      bar = ifelse(clipped, sign(estimate) * cap, estimate),
      lower = ifelse(clipped, bar, estimate - 1.96 * se),
      upper = ifelse(clipped, bar, estimate + 1.96 * se),
      clip_label = ifelse(clipped, paste0(ifelse(estimate > 0, "+", ""), format(round(estimate), big.mark = ",")), ""),
      candidate = term %in% OUTLINE_CANDIDATES & n_significant == length(STRATEGIES),
      outlined = candidate & abs(estimate) == max(abs(estimate)[candidate], -Inf),
      star_y = max(upper) * 1.10
    )
}

panel_plot <- function(data, title, y_label, rotate_labels = FALSE) {
  ggplot(data, aes(term, bar, fill = group)) +
    geom_col(width = 0.72, aes(colour = outlined, linewidth = outlined)) +
    scale_colour_manual(values = c(`TRUE` = "black", `FALSE` = NA), guide = "none") +
    scale_linewidth_manual(values = c(`TRUE` = 1.1, `FALSE` = 0), guide = "none") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25, colour = "grey30") +
    geom_hline(yintercept = 0, colour = "grey50") +
    geom_text(aes(y = star_y, label = stars), size = 4.5) +
    geom_text(aes(y = bar * 0.5, label = clip_label), size = 3.5, angle = 90, colour = "white", fontface = "bold") +
    scale_fill_manual(values = GROUP_COLOURS, name = NULL) +
    scale_x_discrete(labels = LABELS[match(levels(droplevels(data$term)), PREDICTORS)], drop = TRUE) +
    guides(fill = guide_legend(nrow = 1)) +
    labs(x = NULL, y = y_label, title = title) +
    theme_bw(base_size = 13) +
    theme(axis.text.x = element_text(size = 12, angle = if (rotate_labels) 30 else 0, hjust = if (rotate_labels) 1 else 0.5),
          legend.position = "bottom",
          plot.title = element_text(face = "bold", size = 12), panel.grid.major.x = element_blank())
}

make_figure <- function(with_controls) {
  top <- panel_plot(panel_data(estimates, "Natural Stories", with_controls), "Natural Stories", "ms per SD", with_controls)
  bottom <- panel_plot(panel_data(estimates, "CLASP", with_controls), "CLASP", "rating points per SD", with_controls)
  (top / bottom) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
}

for (with_controls in c(FALSE, TRUE)) {
  stem <- if (with_controls) "fig_estimates_controls" else "fig_estimates"
  figure <- make_figure(with_controls)
  ggsave(file.path(FIGURES_DIR, paste0(stem, ".pdf")), figure, width = 11, height = 6.4)
  ggsave(file.path(FIGURES_DIR, paste0(stem, ".png")), figure, width = 11, height = 6.4, dpi = 170)
  outlined <- bind_rows(panel_data(estimates, "Natural Stories", with_controls), panel_data(estimates, "CLASP", with_controls)) %>%
    filter(outlined)
  print_table(outlined %>% select(dataset, term, estimate, n_significant), paste("outlined bars in", stem))
}

write_csv(bind_rows(panel_data(estimates, "Natural Stories", TRUE), panel_data(estimates, "CLASP", TRUE)) %>%
            select(dataset, term, group, estimate, se, stat, stars, n_significant, outlined),
          file.path(RESULTS_DIR, "figure_estimates_data.csv"))
cat("written\n")
