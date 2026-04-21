suppressPackageStartupMessages({
  library(rCausalMGM)
  library(dplyr)
  library(tibble)
})

GRAPH_ATTR <- list(
  layout = "dot",
  rankdir = "LR",
  ranksep = 1.1,
  nodesep = 0.55
)

NODE_ATTR <- list(
  shape = "box",
  fontsize = 28,
  width = 2.6,
  height = 1.1,
  fixedsize = FALSE,
  margin = 0.22
)

EDGE_ATTR <- list(
  fontsize = 22,
  arrowsize = 1.2,
  penwidth = 1.8,
  labelfloat = FALSE,
  decorate = TRUE,
  labeldistance = 1.4,
  label = "",
  xlabel = "",
  headlabel = "",
  taillabel = ""
)

escape_dot <- function(value) {
  gsub('"', '\\"', as.character(value), fixed = TRUE)
}

strip_edge_strength <- function(value) {
  gsub("\\s*\\([^)]*\\)\\s*$", "", as.character(value))
}

relation_to_dot_attrs <- function(relation) {
  clean <- strip_edge_strength(relation)
  if (clean == "-->") {
    return('dir=forward arrowhead=normal arrowtail=none')
  }
  if (clean == "<--") {
    return('dir=back arrowhead=normal arrowtail=none')
  }
  if (clean == "<->") {
    return('dir=both arrowhead=normal arrowtail=normal')
  }
  if (clean == "o->") {
    return('dir=both arrowhead=normal arrowtail=odot')
  }
  if (clean == "<-o") {
    return('dir=both arrowhead=odot arrowtail=normal')
  }
  if (clean == "o-o") {
    return('dir=both arrowhead=odot arrowtail=odot')
  }
  if (clean == "---") {
    return('dir=none arrowhead=none arrowtail=none')
  }
  if (clean == "--") {
    return('dir=none arrowhead=none arrowtail=none')
  }
  'dir=forward arrowhead=normal arrowtail=none'
}

write_clean_dot_from_sif <- function(sif_path, dot_path, graph_label) {
  edges <- read.table(
    sif_path,
    sep = "\t",
    stringsAsFactors = FALSE,
    quote = "",
    comment.char = "",
    fill = TRUE,
    col.names = c("src", "rel", "dst")
  )
  lines <- c(
    "digraph G {",
    '  graph [layout=dot rankdir=LR ranksep=1.1 nodesep=0.55 labelloc="t"];',
    sprintf('  label="%s";', escape_dot(graph_label)),
    '  node [shape=box fontsize=28 width=2.6 height=1.1 fixedsize=false margin="0.22,0.16"];',
    '  edge [fontsize=22 arrowsize=1.2 penwidth=1.8 label="" xlabel="" headlabel="" taillabel=""];'
  )
  for (idx in seq_len(nrow(edges))) {
    src <- escape_dot(edges$src[[idx]])
    dst <- escape_dot(edges$dst[[idx]])
    attrs <- relation_to_dot_attrs(edges$rel[[idx]])
    lines <- c(lines, sprintf('  "%s" -> "%s" [%s];', src, dst, attrs))
  }
  lines <- c(lines, "}")
  writeLines(lines, dot_path)
}

render_dot_png <- function(dot_path, png_path) {
  dot_bin <- Sys.which("dot")
  if (!nzchar(dot_bin)) {
    return(FALSE)
  }
  status <- suppressWarnings(
    system2(dot_bin, c("-Tpng", dot_path, "-o", png_path))
  )
  identical(status, 0L)
}

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
    fci_sif_path <- file.path(output_dir, paste0(prefix, "_", target, "_a0_05_rCausalMGM.sif"))
    saveGraph(fci_graph, filename = fci_sif_path)
    saveGraph(boot_results, filename = file.path(output_dir, paste0(prefix, "_", target, "_a0_05_bootstrap_rCausalMGM.sif")))
    write.csv(boot_results$stabilities, file.path(output_dir, paste0(prefix, "_", target, "_a0_05_bootstrap_stabilities.csv")), row.names = FALSE)
    plot(
      fci_graph,
      graphAttr = GRAPH_ATTR,
      nodeAttr = NODE_ATTR,
      edgeAttr = EDGE_ATTR,
      main = paste("FCI-Stable Graph for", target)
    )
    plot(
      boot_results,
      graphAttr = GRAPH_ATTR,
      nodeAttr = NODE_ATTR,
      edgeAttr = EDGE_ATTR,
      main = paste("Bootstrapped Ensemble Graph for", target)
    )

    if (identical(target, "Disease.Health.State")) {
      dot_path <- file.path(output_dir, "rcausal_graphviz_neighborhood_disease_state.dot")
      png_path <- file.path(output_dir, "rcausal_graphviz_neighborhood_disease_state.png")
      write_clean_dot_from_sif(
        fci_sif_path,
        dot_path,
        "Neighborhood Disease State Graph"
      )
      if (!render_dot_png(dot_path, png_path)) {
        png(
          png_path,
          width = 3200,
          height = 2200,
          res = 220
        )
        plot(
          fci_graph,
          graphAttr = GRAPH_ATTR,
          nodeAttr = NODE_ATTR,
          edgeAttr = EDGE_ATTR,
          main = "Neighborhood Disease State Graph"
        )
        dev.off()
      }
    }
  }
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(args$input_file) || is.null(args$output_dir)) {
  stop("Expected --input-file and --output-dir.")
}
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

enrichment_data <- read.csv(args$input_file)
enrichment_data <- enrichment_data %>% column_to_rownames("field_of_view")
feature_vars <- grep("^enrichment_", colnames(enrichment_data), value = TRUE)
run_analysis(enrichment_data, feature_vars, args$output_dir, "FOV_Enrichment", args$num_boots)
