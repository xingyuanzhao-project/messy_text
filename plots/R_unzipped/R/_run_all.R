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
  file.path(getwd(), "scripts", "R")
}
R_DIR <- .find_script_dir()

required_pkgs <- c(
  "readr", "dplyr", "tidyr", "stringr", "forcats",
  "ggplot2", "scales", "patchwork", "ggrepel", "systemfonts", "purrr"
)
missing_pkgs <- setdiff(required_pkgs, rownames(installed.packages()))
if (length(missing_pkgs)) {
  message("Installing missing packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
}

source(file.path(R_DIR, "_theme.R"))
source(file.path(R_DIR, "_helpers.R"))
source(file.path(R_DIR, "plot_descriptive.R"))
source(file.path(R_DIR, "plot_benchmarks.R"))
source(file.path(R_DIR, "plot_support.R"))
source(file.path(R_DIR, "plot_six_model_accuracy.R"))

main_descriptive()
main_benchmarks()
main_support()
main_six_model_accuracy()

message("All plots regenerated.")
