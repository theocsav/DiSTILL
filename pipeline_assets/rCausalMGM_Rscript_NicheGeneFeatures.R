suppressPackageStartupMessages({
  library(rCausalMGM)
  library(dplyr)
  library(tibble)
})

parse_args <- function(args) {
  out <- list(input_file = NULL, output_dir = NULL, num_boots = 20)
  idx <- 1
  while (idx <= length(args)) {
    key <- args[[idx]]
    val <- if (idx + 1 <= length(args)) args[[idx + 1]] else NULL
    if (key == "--input-file") out$input_file <- val
    if (key == "--output-dir") out$output_dir <- val
    if (key == "--num-boots") out$num_boots <- as.integer(val)
    idx <- idx + 2
  }
  out
}

ensure_targets <- function(df) {
  if ("Disease/Health State" %in% colnames(df) && !("Disease.Health.State" %in% colnames(df))) {
    df[["Disease.Health.State"]] <- df[["Disease/Health State"]]
  }
  if (!("Disease.Health.State" %in% colnames(df))) {
    stop("Missing Disease.Health.State or Disease/Health State column.")
  }
  states <- unique(as.character(df[["Disease.Health.State"]]))
  states <- states[!is.na(states) & nzchar(states)]
  for (state in states) {
    safe_name <- make.names(state)
    df[[safe_name]] <- factor(ifelse(as.character(df[["Disease.Health.State"]]) == state, state, paste0("not_", safe_name)))
  }
  df[["Disease.Health.State"]] <- factor(df[["Disease.Health.State"]])
  df
}

run_analysis <- function(data_frame, feature_vars, output_dir, prefix, num_boots) {
  data_frame <- ensure_targets(data_frame)
  target_vars <- c("Disease.Health.State", setdiff(colnames(data_frame), c(feature_vars, "field_of_view", "Disease/Health State")))
  pdf(file.path(output_dir, paste0(prefix, "_Combined_Causal_Analysis.pdf")), width = 14, height = 14)
  on.exit(dev.off(), add = TRUE)

  for (target in target_vars) {
    if (!(target %in% colnames(data_frame))) next
    subset_data <- data_frame %>% select(any_of(feature_vars), any_of(target))
    if (!(target %in% colnames(subset_data)) || length(unique(subset_data[[target]])) <= 1) {
      next
    }
    fci_graph <- fciStable(data = subset_data, orientRule = "maxp", alpha = 0.05, verbose = TRUE)
    boot_results <- bootstrap(data = subset_data, graph = fci_graph, numBoots = num_boots, threads = -1L, verbose = TRUE)
    saveGraph(fci_graph, filename = file.path(output_dir, paste0(prefix, "_", target, "_a0_05_rCausalMGM.sif")))
    saveGraph(boot_results, filename = file.path(output_dir, paste0(prefix, "_", target, "_a0_05_bootstrap_rCausalMGM.sif")))
    write.csv(boot_results$stabilities, file.path(output_dir, paste0(prefix, "_", target, "_a0_05_bootstrap_stabilities.csv")), row.names = FALSE)
    plot(fci_graph, nodeAttr = list(fontsize = 20), main = paste("FCI-Stable Graph for", target))
    plot(boot_results, nodeAttr = list(fontsize = 20), main = paste("Bootstrapped Ensemble Graph for", target))
  }
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(args$input_file) || is.null(args$output_dir)) {
  stop("Expected --input-file and --output-dir.")
}
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

niche_gene_data <- read.csv(args$input_file)
niche_gene_data <- niche_gene_data %>% column_to_rownames("field_of_view")
feature_vars <- setdiff(colnames(niche_gene_data), c("Disease/Health State", "Disease.Health.State"))
run_analysis(niche_gene_data, feature_vars, args$output_dir, "NicheGene", args$num_boots)
