suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(forcats)
  library(purrr)
  library(ggplot2)
  library(patchwork)
  library(ggrepel)
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

CONFIGS <- list(
  less_labels_multi_models = list(
    name = "less_labels_multi_models",
    labels = c("desenlace", "vic_grupo_social", "captura_tipo"),
    thresholds = c(1.0, 0.6),
    input = file.path(
      PROJECT_ROOT, "results_down_sized",
      "df_text_by_report_classification_consolidated_eval_supported.csv"
    ),
    output_dir = file.path(
      PROJECT_ROOT, "plots", "evaluation", "matching", "less_labels_multi_models"
    )
  ),
  all_labels_one_model = list(
    name = "all_labels_one_model",
    labels = c(
      "vic_grupo_social", "amenaza_quien", "captura_metodo", "captura_tipo",
      "cautiverio_trato", "desenlace", "desenlace_tipo", "perp_tipo1",
      "perp_tipo2", "proced_contacto1", "proced_contacto2", "proced_contactado",
      "Tribunal_tipo", "proced_sent_tipo", "soc_civil"
    ),
    thresholds = c(1.0, 0.75, 0.50, 0.25, 0.10),
    input = file.path(
      PROJECT_ROOT,
      "df_text_by_report_conversation_classification (2)_eval_supported.csv"
    ),
    output_dir = file.path(
      PROJECT_ROOT, "plots", "evaluation", "matching", "all_labels_one_model"
    )
  )
)

threshold_label <- function(frac) if (frac >= 1) "Full" else paste0("≥ ", round(frac * 100), "%")
threshold_count <- function(frac, n_labels) as.integer(ceiling(frac * n_labels))

resolve_thresholds <- function(cfg) {
  fracs <- sort(unique(c(cfg$thresholds, 1.0)), decreasing = TRUE)
  tibble(
    fraction = fracs,
    count    = vapply(fracs, threshold_count, integer(1), n_labels = length(cfg$labels)),
    label    = vapply(fracs, threshold_label, character(1)),
    linetype = c("solid", "longdash", "dashed", "dotdash", "dotted", "twodash")[seq_along(fracs)]
  )
}

first_turn_at_threshold <- function(df, threshold_count) {
  all_pairs <- df |> distinct(victim, model)
  reached <- df |>
    filter(support_count >= threshold_count) |>
    group_by(victim, model) |>
    summarise(first_turn = min(turn_index), .groups = "drop")
  all_pairs |> left_join(reached, by = c("victim", "model"))
}

build_per_label_support_heatmap <- function(df, cfg) {
  supported_cols <- paste0(cfg$labels, "_supported")

  ever_supported <- df |>
    group_by(victim, model) |>
    summarise(
      across(all_of(supported_cols), ~ as.integer(max(.x, na.rm = TRUE) > 0)),
      .groups = "drop"
    ) |>
    pivot_longer(all_of(supported_cols),
                 names_to = "label_col", values_to = "supported") |>
    mutate(label = sub("_supported$", "", label_col))

  heatmap_data <- ever_supported |>
    group_by(model, label) |>
    summarise(
      frac        = mean(supported, na.rm = TRUE),
      n_supported = sum(supported,  na.rm = TRUE),
      n_total     = n(),
      .groups     = "drop"
    ) |>
    mutate(
      model_display = factor(
        unname(MODEL_DISPLAY[model]) %||% model,
        levels = rev(c("Llama 8B", "Gemma 12B", "Ministral 8B"))
      ),
      label_display = factor(
        unname(LABEL_DISPLAY[label]) %||% label,
        levels = unname(LABEL_DISPLAY[cfg$labels])
      ),
      cell_label = sprintf("%.0f%%\n(%s)", 100 * frac, fmt_n(n_supported))
    )

  per_model_summary <- heatmap_data |>
    group_by(model_display) |>
    summarise(model_avg = mean(frac), .groups = "drop")

  range_text <- heatmap_data |>
    group_by(label_display) |>
    summarise(
      spread = max(frac) - min(frac),
      .groups = "drop"
    )

  n_pairs_total <- heatmap_data |> distinct(model, n_total) |> pull(n_total) |> sum()
  subtitle <- paste0(
    "% of (victim, model) pairs with label supported at any turn · ",
    fmt_n(n_pairs_total), " pairs"
  )

  ggplot(heatmap_data,
         aes(x = label_display, y = model_display, fill = frac)) +
    geom_tile(color = "white", linewidth = 1.5) +
    geom_text(
      aes(label = cell_label, color = frac > 0.55),
      family = paper_font(), size = 3.4, fontface = "plain", lineheight = 0.95
    ) +
    scale_fill_gradient(
      low      = PALETTE_BG_LIGHT,
      high     = PALETTE_BG_DARK,
      labels   = label_percent(accuracy = 1),
      limits   = c(0, 1),
      breaks   = c(0, 0.25, 0.5, 0.75, 1.0),
      guide    = guide_colorbar(
        barheight = unit(0.35, "lines"),
        barwidth  = unit(8,    "lines"),
        ticks     = FALSE,
        title.position = "top"
      )
    ) +
    scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "grey15"),
                       guide = "none") +
    scale_x_discrete(position = "top") +
    labs(
      subtitle = subtitle,
      x = NULL, y = NULL,
      fill = "Coverage"
    ) +
    theme_paper() +
    theme(
      panel.grid       = element_blank(),
      axis.text.x.top  = element_text(size = 10, color = "grey15", margin = margin(b = 4)),
      axis.text.y      = element_text(size = 10, color = "grey15", face = "bold"),
      legend.position  = "bottom",
      legend.title     = element_text(size = 9, color = "grey25"),
      legend.text      = element_text(size = 9, color = "grey25"),
      plot.subtitle    = element_text(margin = margin(b = 10))
    )
}

build_ceiling_panel <- function(df, cfg) {
  human_cols <- cfg$labels
  per_victim <- df |>
    distinct(victim, .keep_all = TRUE)
  per_victim$n_human_labels <- rowSums(!is.na(per_victim[, human_cols, drop = FALSE]) &
                                          per_victim[, human_cols, drop = FALSE] != "")

  max_support <- df |>
    group_by(victim) |>
    summarise(max_support = max(support_count, na.rm = TRUE), .groups = "drop")

  per_victim <- per_victim |>
    left_join(max_support, by = "victim") |>
    mutate(fully_supported = max_support >= n_human_labels)

  excluded <- sum(per_victim$n_human_labels == 0)
  annotated <- per_victim |> filter(n_human_labels > 0)
  total_ann <- nrow(annotated)
  total_full <- sum(annotated$fully_supported)

  summary_df <- annotated |>
    group_by(n_human_labels) |>
    summarise(
      full    = sum(fully_supported),
      partial = sum(!fully_supported),
      .groups = "drop"
    ) |>
    pivot_longer(c(full, partial), names_to = "coverage", values_to = "n") |>
    mutate(
      coverage = factor(
        coverage,
        levels = c("full", "partial"),
        labels = c("Annotations fully covered", "Annotations partially covered")
      ),
      n_label = if_else(n_human_labels == 1L,
                        paste0(n_human_labels, " label"),
                        paste0(n_human_labels, " labels")),
      n_label = factor(n_label, levels = unique(n_label[order(n_human_labels)]))
    )

  subtitle <- paste0(
    "victims with annotations: ", fmt_n(total_ann),
    " · full ", fmt_pct(total_full / total_ann),
    " · partial ", fmt_pct((total_ann - total_full) / total_ann),
    " · excluded (no labels): ", fmt_n(excluded)
  )

  ggplot(summary_df, aes(x = n_label, y = n, fill = coverage)) +
    geom_col(
      position = position_dodge2(preserve = "single", padding = 0.1),
      width = 0.75
    ) +
    geom_text(
      aes(label = fmt_n(n)),
      position = position_dodge2(width = 0.75, preserve = "single"),
      vjust = -0.5, family = paper_font(), size = 2.9, color = "grey20"
    ) +
    scale_fill_manual(values = c(
      "Annotations fully covered"     = PALETTE_ACCENT_POS,
      "Annotations partially covered" = PALETTE_NEUTRAL
    )) +
    scale_y_continuous(
      labels = label_comma(),
      expand = expansion(mult = c(0, 0.12))
    ) +
    labs(
      subtitle = subtitle,
      x = "Number of human annotations per victim",
      y = "Number of victims"
    ) +
    theme_paper() +
    theme(legend.position = "bottom")
}

run_config <- function(cfg) {
  if (!file.exists(cfg$input)) {
    warning("Input not found, skipping ", cfg$name, ": ", cfg$input)
    return(invisible(NULL))
  }
  df <- read_csv(cfg$input, show_col_types = FALSE,
                 guess_max = 200000)

  required <- c("victim", "model", "turn_index", "support_count", cfg$labels)
  missing  <- setdiff(required, colnames(df))
  if (length(missing)) {
    warning("Skipping ", cfg$name, " — missing columns: ", paste(missing, collapse = ", "))
    return(invisible(NULL))
  }

  single_model <- length(unique(df$model)) == 1
  p_ceil <- build_ceiling_panel(df, cfg)

  if (single_model) {
    ggsave_paper(
      p_ceil,
      file.path(cfg$output_dir, "joint_cdf_and_annotation_coverage.png"),
      width = 6.5, height = 3
    )
  } else {
    p_top <- build_per_label_support_heatmap(df, cfg)
    joint <- (p_top / p_ceil) + plot_layout(heights = c(0.85, 1.15))
    ggsave_paper(
      joint,
      file.path(cfg$output_dir, "joint_cdf_and_annotation_coverage.png"),
      width = 7, height = 5.0
    )
  }
}

main_support <- function() {
  for (cfg in CONFIGS) run_config(cfg)
}

if (sys.nframe() == 0) main_support()
