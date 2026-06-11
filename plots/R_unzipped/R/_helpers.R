suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(forcats)
})

fmt_n  <- function(x) format(x, big.mark = ",", scientific = FALSE)
fmt_pct <- function(x, digits = 0) paste0(formatC(100 * x, digits = digits, format = "f"), "%")

num_stats_subtitle <- function(values, units = "") {
  v <- values[!is.na(values)]
  n <- length(v)
  mu <- mean(v); med <- median(v); mn <- min(v); mx <- max(v)
  parts <- c(
    paste0("n = ", fmt_n(n)),
    paste0("mean ", fmt_n(round(mu)), if (nchar(units)) paste0(" ", units) else ""),
    paste0("median ", fmt_n(round(med)), if (nchar(units)) paste0(" ", units) else ""),
    paste0("range ", fmt_n(round(mn)), "–", fmt_n(round(mx)))
  )
  paste(parts, collapse = " · ")
}

cat_stats_subtitle <- function(values, label_map = NULL) {
  total <- length(values)
  present <- sum(!is.na(values) & nzchar(values))
  missing <- total - present
  distinct <- length(unique(values[!is.na(values) & nzchar(values)]))
  v <- values[!is.na(values) & nzchar(values)]
  tab <- sort(table(v), decreasing = TRUE)
  top_name <- if (length(tab)) {
    raw <- names(tab)[1]
    if (!is.null(label_map) && !is.na(label_map[raw])) unname(label_map[raw]) else raw
  } else NA_character_
  top <- if (length(tab)) paste0("top: ", top_name, " (", fmt_n(unname(tab)[1]), ")") else "top: —"
  parts <- c(
    paste0("n = ", fmt_n(present), " victims"),
    paste0("missing ", fmt_n(missing), " (", fmt_pct(missing / max(total, 1)), ")"),
    paste0(distinct, " distinct"),
    top
  )
  paste(parts, collapse = " · ")
}

ggsave_paper <- function(plot, path, width, height) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggsave(
    filename = path, plot = plot,
    width = width, height = height, units = "in",
    dpi = 300, bg = "white"
  )
  message("Wrote ", path)
  invisible(path)
}

short_model <- function(x) {
  out <- unname(MODEL_DISPLAY[x])
  ifelse(is.na(out), basename(x), out)
}
