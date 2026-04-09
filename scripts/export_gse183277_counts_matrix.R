args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("usage: Rscript export_gse183277_counts_matrix.R <counts_rds_like_file> <out_prefix>")
}

counts_file <- args[[1]]
out_prefix <- args[[2]]

suppressPackageStartupMessages({
  library(Matrix)
})

obj <- readRDS(counts_file)
if (!(inherits(obj, "Matrix") || is.matrix(obj))) {
  stop("counts object is not a matrix-like object")
}

if (!inherits(obj, "dgCMatrix")) {
  obj <- as(obj, "dgCMatrix")
}

dir.create(dirname(out_prefix), recursive = TRUE, showWarnings = FALSE)

matrix_path <- paste0(out_prefix, ".mtx")
genes_path <- paste0(out_prefix, "_genes.csv")
barcodes_path <- paste0(out_prefix, "_barcodes.csv")

Matrix::writeMM(obj, matrix_path)
write.csv(data.frame(gene = rownames(obj)), genes_path, row.names = FALSE)
write.csv(data.frame(cell_barcode = colnames(obj)), barcodes_path, row.names = FALSE)

cat("matrix=", matrix_path, "\n", sep = "")
cat("genes=", genes_path, "\n", sep = "")
cat("barcodes=", barcodes_path, "\n", sep = "")
