# plot_six_model_accuracy.R
# 6-model accuracy comparison on the 100-victim subsample
# Inputs:  comparison_run/metrics_full.csv  (long format: model, field, n, accuracy, precision, recall, f1, kappa)
# Outputs: plots/evaluation/benchmarks/six_model_accuracy.png

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(forcats)
  library(stringr)
  library(scales)
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

PROJECT_ROOT      <- normalizePath(file.path(R_DIR, "..", ".."))
SIX_MODEL_INPUT   <- file.path(PROJECT_ROOT, "comparison_run", "metrics_full.csv")
SIX_MODEL_OUT_DIR <- file.path(PROJECT_ROOT, "plots", "evaluation", "benchmarks")

FIELD_LABELS <- c(
  desenlace        = "Outcome of the disappearance",
  vic_grupo_social = "Social group membership",
  captura_tipo     = "Type of place of disappearance"
)

FAMILY_LEVELS <- c("Open-source (vLLM)", "Closed-source (Claude 4.x)")
FAMILY_FILL   <- c(
  "Open-source (vLLM)"         = PALETTE_ACCENT_POS,
  "Closed-source (Claude 4.x)" = PALETTE_NEUTRAL
)

MODEL_ORDER <- c(
  "Llama-3.1-8B", "Ministral-3-8B", "Gemma-3-12B",
  "Claude-Haiku-4.5", "Claude-Opus-4.7", "Claude-Sonnet-4.6"
)

build_six_model_plot <- function(df) {
  df <- df %>%
    mutate(
      family    = factor(
        if_else(str_starts(model, "Claude"),
                "Closed-source (Claude 4.x)",
                "Open-source (vLLM)"),
        levels = FAMILY_LEVELS
      ),
      field_lab = factor(FIELD_LABELS[field], levels = unname(FIELD_LABELS)),
      model     = factor(model, levels = MODEL_ORDER)
    )

  n_min <- min(df$n)
  n_max <- max(df$n)
  n_text <- if (n_min == n_max) {
    paste0("n = ", fmt_n(n_min),
           " victims per field · dashed = chance · dotted = open/closed boundary")
  } else {
    paste0("n = ", fmt_n(n_min), "–", fmt_n(n_max),
           " victims per field · dashed = chance · dotted = open/closed boundary")
  }

  ggplot(df, aes(x = model, y = accuracy, fill = family)) +
    geom_hline(yintercept = 0.5, linetype = "dashed",
               color = "grey60", linewidth = 0.4) +
    geom_vline(xintercept = 3.5, linetype = "dotted",
               color = "grey55", linewidth = 0.4) +
    geom_col(width = 0.65) +
    geom_text(
      aes(label = sprintf("%.2f", accuracy)),
      vjust = -0.5, family = paper_font(), size = 2.9, color = "grey20"
    ) +
    facet_wrap(~ field_lab, nrow = 1) +
    scale_fill_manual(values = FAMILY_FILL) +
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
      legend.position    = "bottom",
      panel.spacing.x    = unit(1.2, "lines"),
      strip.text         = element_text(face = "plain", color = "grey15"),
      axis.text.x        = element_text(angle = 30, hjust = 1)
    )
}

main_six_model_accuracy <- function() {
  if (!file.exists(SIX_MODEL_INPUT)) {
    warning("Skipping six_model_accuracy — input not found: ", SIX_MODEL_INPUT)
    return(invisible(NULL))
  }
  df <- read_csv(SIX_MODEL_INPUT, show_col_types = FALSE)
  p  <- build_six_model_plot(df)
  ggsave_paper(p, file.path(SIX_MODEL_OUT_DIR, "six_model_accuracy.png"),
               width = 9.5, height = 4.0)
}

if (sys.nframe() == 0) main_six_model_accuracy()
