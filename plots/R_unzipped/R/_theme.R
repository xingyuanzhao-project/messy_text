suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

PAPER_FONT_SERIF <- "Source Serif Pro"
PAPER_FONT_FALLBACK <- "serif"

paper_font <- function() {
  fams <- tryCatch(systemfonts::system_fonts()$family, error = function(e) character())
  if (PAPER_FONT_SERIF %in% fams) PAPER_FONT_SERIF else PAPER_FONT_FALLBACK
}

palette_paper <- function() {
  c(
    "Llama 8B"     = "#95c36e",
    "Gemma 12B"    = "#74a08b",
    "Ministral 8B" = "#6c5d7c",
    "Average"      = "#2c2f34"
  )
}

palette_paper_seq <- function(n = 5) {
  ramp <- c("#cfe1d6", "#95c36e", "#74a08b", "#4f7766", "#2f4858")
  if (n <= length(ramp)) ramp[seq_len(n)] else colorRampPalette(ramp)(n)
}

PALETTE_NEUTRAL    <- "#6c5d7c"
PALETTE_ACCENT_POS <- "#74a08b"
PALETTE_ACCENT_NEG <- "#c1666b"
PALETTE_BG_LIGHT   <- "#cfe1d6"
PALETTE_BG_DARK    <- "#2f4858"
PALETTE_RULE_MEAN  <- "#2c2f34"
PALETTE_RULE_MED   <- "#6c5d7c"

MODEL_DISPLAY <- c(
  "mistralai/Ministral-3-8B-Instruct-2512"             = "Ministral 8B",
  "valhalla/ministral-8b-instruct-v0.3-awq"            = "Ministral 8B",
  "gaunernst/gemma-3-12b-it-int4-awq"                  = "Gemma 12B",
  "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" = "Llama 8B"
)

LABEL_DISPLAY <- c(
  desenlace        = "Outcome of the disappearance",
  captura_tipo     = "Type of place of disappearance",
  vic_grupo_social = "Social group membership"
)

theme_paper <- function(base_size = 10) {
  fam <- paper_font()
  theme_minimal(base_size = base_size, base_family = fam) +
    theme(
      plot.title           = element_text(size = base_size + 1, face = "plain", margin = margin(b = 4)),
      plot.subtitle        = element_text(size = base_size - 1, color = "grey35", margin = margin(b = 8)),
      plot.caption         = element_text(size = base_size - 2, color = "grey45", hjust = 0),
      plot.title.position  = "plot",
      plot.caption.position = "plot",
      panel.grid.minor     = element_blank(),
      panel.grid.major.y   = element_line(color = "grey92", linewidth = 0.3),
      panel.grid.major.x   = element_blank(),
      axis.line            = element_blank(),
      axis.ticks           = element_blank(),
      axis.title           = element_text(size = base_size - 1, color = "grey25"),
      axis.text            = element_text(size = base_size - 1, color = "grey20"),
      strip.text           = element_text(face = "italic", size = base_size, color = "grey20"),
      legend.position      = "bottom",
      legend.title         = element_blank(),
      legend.text          = element_text(size = base_size - 1),
      legend.key.height    = unit(0.4, "lines"),
      plot.margin          = margin(8, 12, 8, 8)
    )
}

theme_paper_flipped <- function(base_size = 10) {
  theme_paper(base_size) +
    theme(
      panel.grid.major.x = element_line(color = "grey92", linewidth = 0.3),
      panel.grid.major.y = element_blank()
    )
}

scale_x_count <- function(...) scale_x_continuous(labels = label_comma(), ...)
scale_y_count <- function(...) scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.08)), ...)
scale_y_pct   <- function(...) scale_y_continuous(labels = label_percent(accuracy = 1), ...)
