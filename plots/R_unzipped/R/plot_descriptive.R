suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(forcats)
  library(ggplot2)
  library(patchwork)
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

PROJECT_ROOT <- normalizePath(file.path(R_DIR, "..", ".."))
DESC_INPUT_CSV    <- file.path(PROJECT_ROOT, "df_text_by_report.csv")
DESC_OUTPUT_DIR   <- file.path(PROJECT_ROOT, "plots", "descriptive_statistics")

SHORT_LABELS <- list(
  desenlace = c(
    "Still disappeared"                                = "Still disappeared",
    "Liberated by captors"                             = "Lib. by captors",
    "Liberated by authorities"                         = "Lib. by authorities",
    "Found dead"                                       = "Found dead",
    "Escaped or was liberated through their own means" = "Escaped / self-liberated",
    "Found alive"                                      = "Found alive",
    "Found, but does not specify if dead or alive"     = "Found, unspecified"
  ),
  vic_grupo_social = c(
    "Professionals (Entrepreneur, Engineer, Professor, Journalist, etc)"  = "Professionals",
    "People that work in service industries (taxi driver, salesman, etc)" = "Service workers",
    "Civil servants (Police, mayor, public worker, etc)"                  = "Civil servants",
    "Belonging to some sexual identity group (LGBTQ)"                     = "LGBTQ",
    "People associated with politics"                                     = "Politics",
    "Activists (political activist, human rights, etc)"                   = "Activists",
    "Organized crime"                                                     = "Organized crime",
    "Students"                                                            = "Students",
    "Land Worker"                                                         = "Land workers",
    "Other"                                                               = "Other"
  ),
  captura_tipo = c(
    "Places related to the victim (house, workplace, private property)" = "Victim-related places",
    "Economic, social, industrial, agricultural and service centers"    = "Economic / social centers",
    "Authorities (government offices, military facilities)"             = "Authorities",
    "Educational and medical facilities"                                = "Educational / medical",
    "Places for free expression, association and gatherings"            = "Expression / gatherings",
    "Unoccupied or barren public spaces"                                = "Open public spaces",
    "Means and routes of transport and places of connection"            = "Transport / connection",
    "International and protected spaces"                                = "International / protected",
    "Special centers and barracks for detention"                        = "Detention centers"
  )
)

`%||%` <- function(a, b) if (is.null(a)) b else a

normalize_value <- function(x) {
  x <- ifelse(is.na(x), NA_character_, trimws(as.character(x)))
  x <- ifelse(nzchar(x) & x != "No information", x, NA_character_)
  x
}

victim_consensus <- function(df, col) {
  values <- normalize_value(df[[col]])
  tibble(victim = df$victim, value = values) |>
    filter(!is.na(value)) |>
    group_by(victim, value) |>
    summarise(n = n(), .groups = "drop_last") |>
    arrange(desc(n), value, .by_group = TRUE) |>
    slice(1) |>
    ungroup() |>
    select(victim, !!col := value)
}

build_victim_labels <- function(df) {
  all_victims <- tibble(victim = unique(df$victim))
  Reduce(
    function(acc, col) left_join(acc, victim_consensus(df, col), by = "victim"),
    names(SHORT_LABELS),
    init = all_victims
  )
}

build_corpus_distributions <- function(df) {
  per_victim <- df |>
    group_by(victim) |>
    summarise(
      n_docs = n(),
      median_chars = median(text_len, na.rm = TRUE),
      .groups = "drop"
    )

  n_victims <- nrow(per_victim)
  n_docs_total <- nrow(df)
  med_chars_overall <- median(df$text_len, na.rm = TRUE)
  med_docs <- median(per_victim$n_docs)

  x_breaks <- pretty(per_victim$n_docs, n = 8)
  y_breaks <- c(100, 300, 1000, 3000, 10000, 30000, 100000)

  scatter <- ggplot(per_victim, aes(x = n_docs, y = median_chars)) +
    geom_jitter(width = 0.18, height = 0,
                color = PALETTE_NEUTRAL, alpha = 0.45, size = 1.3, stroke = 0) +
    geom_hline(yintercept = med_chars_overall,
               color = "grey60", linetype = "dashed", linewidth = 0.4) +
    geom_vline(xintercept = med_docs,
               color = "grey60", linetype = "dashed", linewidth = 0.4) +
    annotate("text", x = med_docs, y = max(y_breaks),
             label = paste0(" median ", fmt_n(med_docs), " docs"),
             hjust = 0, vjust = 1, family = paper_font(),
             size = 2.9, color = "grey35") +
    annotate("text", x = max(per_victim$n_docs), y = med_chars_overall,
             label = paste0("median ", fmt_n(round(med_chars_overall)), " chars "),
             hjust = 1, vjust = -0.4, family = paper_font(),
             size = 2.9, color = "grey35") +
    scale_x_continuous(breaks = x_breaks, expand = expansion(mult = c(0.02, 0.04))) +
    scale_y_log10(breaks = y_breaks, labels = label_comma(),
                  expand = expansion(mult = c(0.02, 0.04))) +
    labs(
      x = "Documents per victim",
      y = "Median characters per report (log scale)",
      subtitle = paste0(
        fmt_n(n_victims), " victims · ", fmt_n(n_docs_total), " reports · ",
        "median ", fmt_n(med_docs), " docs/victim · ",
        "median ", fmt_n(round(med_chars_overall)), " chars/report"
      )
    ) +
    theme_paper() +
    theme(plot.margin = margin(2, 2, 4, 4))

  top_density <- ggplot(per_victim, aes(x = n_docs)) +
    geom_density(fill = PALETTE_NEUTRAL, color = NA, alpha = 0.55, adjust = 1.1) +
    scale_x_continuous(breaks = x_breaks, expand = expansion(mult = c(0.02, 0.04)),
                       limits = range(x_breaks)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.04))) +
    theme_paper() +
    theme(
      axis.title       = element_blank(),
      axis.text        = element_blank(),
      panel.grid       = element_blank(),
      plot.subtitle    = element_blank(),
      plot.margin      = margin(4, 2, 0, 4)
    )

  right_density <- ggplot(per_victim, aes(x = median_chars)) +
    geom_density(fill = PALETTE_NEUTRAL, color = NA, alpha = 0.55, adjust = 1.1) +
    scale_x_log10(breaks = y_breaks, expand = expansion(mult = c(0.02, 0.04))) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.04))) +
    coord_flip() +
    theme_paper() +
    theme(
      axis.title       = element_blank(),
      axis.text        = element_blank(),
      panel.grid       = element_blank(),
      plot.subtitle    = element_blank(),
      plot.margin      = margin(2, 4, 4, 0)
    )

  (top_density + plot_spacer() + scatter + right_density) +
    plot_layout(
      ncol = 2, nrow = 2,
      widths = c(5, 1), heights = c(1, 5)
    )
}

build_category_bar <- function(victim_df, label_col, title_text = NULL) {
  short <- SHORT_LABELS[[label_col]]
  raw <- victim_df[[label_col]]
  d <- tibble(value = raw[!is.na(raw)]) |>
    mutate(short = unname(short[value])) |>
    filter(!is.na(short)) |>
    count(short, name = "n") |>
    arrange(desc(n)) |>
    mutate(short = fct_inorder(short))

  subtitle <- cat_stats_subtitle(raw, label_map = short)

  ggplot(d, aes(x = n, y = fct_rev(short))) +
    geom_col(fill = PALETTE_NEUTRAL, width = 0.7) +
    geom_text(
      aes(label = fmt_n(n)),
      hjust = -0.2, family = paper_font(), size = 2.9, color = "grey20"
    ) +
    scale_x_continuous(
      labels = label_comma(),
      expand = expansion(mult = c(0, 0.12))
    ) +
    labs(
      x = "Count",
      y = NULL,
      subtitle = subtitle
    ) +
    theme_paper_flipped()
}

main_descriptive <- function() {
  if (!file.exists(DESC_INPUT_CSV)) stop("Input CSV not found: ", DESC_INPUT_CSV)
  df <- read_csv(DESC_INPUT_CSV, show_col_types = FALSE)

  if (!all(c("victim", "text_len", names(SHORT_LABELS)) %in% colnames(df))) {
    stop("Input CSV missing required columns.")
  }

  corpus_plot <- build_corpus_distributions(df)
  ggsave_paper(
    corpus_plot,
    file.path(DESC_OUTPUT_DIR, "corpus_distributions.png"),
    width = 6.5, height = 4.9
  )

  victim_df <- build_victim_labels(df)

  ggsave_paper(
    build_category_bar(victim_df, "desenlace"),
    file.path(DESC_OUTPUT_DIR, "desenlace_distribution_by_victim_before_merging.png"),
    width = 5.2, height = 2.1
  )

  ggsave_paper(
    build_category_bar(victim_df, "captura_tipo"),
    file.path(DESC_OUTPUT_DIR, "captura_tipo_distribution_by_victim_before_merging.png"),
    width = 5.2, height = 2.3
  )

  ggsave_paper(
    build_category_bar(victim_df, "vic_grupo_social"),
    file.path(DESC_OUTPUT_DIR, "vic_grupo_social_distribution_by_victim_before_merging.png"),
    width = 5.2, height = 2.6
  )
}

if (sys.nframe() == 0) main_descriptive()
