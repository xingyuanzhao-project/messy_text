suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(forcats)
  library(ggplot2)
})

.find_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  frames <- sys.frames()
  for (i in seq_along(frames)) {
    of <- frames[[i]]$ofile
    if (!is.null(of) && nzchar(of)) return(dirname(normalizePath(of)))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      nzchar(rstudioapi::getActiveDocumentContext()$path)) {
    return(dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path)))
  }
  file.path(getwd(), "scripts", "R")
}
R_DIR <- .find_script_dir()
source(file.path(R_DIR, "_theme.R"))
source(file.path(R_DIR, "_helpers.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

PROJECT_ROOT <- normalizePath(file.path(R_DIR, "..", ".."))
BENCH_INPUT_DIR    <- file.path(PROJECT_ROOT, "results_down_sized", "evaluation_diffs")
BENCH_OUTPUT_DIR   <- file.path(PROJECT_ROOT, "plots", "evaluation", "benchmarks", "less_labels_multi_models")

MODEL_ORDER_DISPLAY <- c("Llama 8B", "Gemma 12B", "Ministral 8B")
COND_LEVELS <- c("Simple summarization", "Extractive summarization")
COND_FILL <- c("Simple summarization" = PALETTE_NEUTRAL, "Extractive summarization" = PALETTE_ACCENT_POS)

load_benchmark <- function() {
  old <- read_csv(file.path(BENCH_INPUT_DIR, "benchmarks_old_by_model_and_label.csv"), show_col_types = FALSE)
  new <- read_csv(file.path(BENCH_INPUT_DIR, "benchmarks_new_by_model_and_label.csv"), show_col_types = FALSE)
  diffs <- read_csv(file.path(BENCH_INPUT_DIR, "diffs_by_model_and_label.csv"), show_col_types = FALSE)

  long <- bind_rows(
    old |> mutate(condition = "Simple summarization"),
    new |> mutate(condition = "Extractive summarization")
  ) |>
    mutate(
      model_display = unname(MODEL_DISPLAY[model]),
      model_display = factor(model_display, levels = MODEL_ORDER_DISPLAY),
      condition = factor(condition, levels = COND_LEVELS)
    )

  diffs <- diffs |>
    mutate(model_display = factor(unname(MODEL_DISPLAY[model]), levels = MODEL_ORDER_DISPLAY))

  list(long = long, diffs = diffs)
}

build_combined_benchmark_plot <- function(data) {
  label_order <- c("desenlace", "captura_tipo", "vic_grupo_social")
  long <- data$long |>
    filter(label %in% label_order) |>
    mutate(label_display = factor(unname(LABEL_DISPLAY[label]),
                                  levels = unname(LABEL_DISPLAY[label_order])))
  diffs <- data$diffs |>
    filter(label %in% label_order) |>
    mutate(label_display = factor(unname(LABEL_DISPLAY[label]),
                                  levels = unname(LABEL_DISPLAY[label_order])))

  delta_y <- 0.95

  n_min <- min(long$rows)
  n_max <- max(long$rows)
  n_text <- if (n_min == n_max) {
    paste0("n = ", fmt_n(n_min), " victims · italic deltas above bars · dashed = chance")
  } else {
    paste0("n = ", fmt_n(n_min), "–", fmt_n(n_max),
           " victims (varies by model) · italic deltas above bars · dashed = chance")
  }

  ggplot(long, aes(x = model_display, y = Accuracy, fill = condition)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60", linewidth = 0.4) +
    geom_col(position = position_dodge(width = 0.65), width = 0.55) +
    geom_text(
      aes(label = sprintf("%.2f", Accuracy)),
      position = position_dodge(width = 0.65),
      vjust = -0.5, family = paper_font(), size = 2.9, color = "grey20"
    ) +
    geom_text(
      data = diffs,
      aes(x = model_display, y = delta_y,
          label = sprintf("%+0.2f", `Delta Accuracy`)),
      inherit.aes = FALSE,
      family = paper_font(), fontface = "italic", size = 2.9,
      color = "grey25"
    ) +
    facet_wrap(~ label_display, nrow = 1) +
    scale_fill_manual(values = COND_FILL) +
    scale_y_continuous(
      limits = c(0, 1.05),
      breaks = seq(0, 1, 0.2),
      labels = label_percent(accuracy = 1),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      subtitle = n_text,
      x = NULL, y = "Accuracy"
    ) +
    theme_paper() +
    theme(
      legend.position = "bottom",
      panel.spacing.x = unit(1.2, "lines"),
      strip.text      = element_text(face = "plain", color = "grey15"),
      axis.text.x     = element_text(angle = 30, hjust = 1)
    )
}

main_benchmarks <- function() {
  data <- load_benchmark()
  p <- build_combined_benchmark_plot(data)
  ggsave_paper(
    p,
    file.path(BENCH_OUTPUT_DIR, "metrics_combined.png"),
    width = 9.5, height = 4.0
  )
}

if (sys.nframe() == 0) main_benchmarks()
