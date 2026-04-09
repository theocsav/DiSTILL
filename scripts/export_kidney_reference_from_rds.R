args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("usage: Rscript export_kidney_reference_from_rds.R <ref_rds> <out_matrix_mtx> <out_obs_csv>")
}

ref_rds <- args[[1]]
out_matrix <- args[[2]]
out_obs <- args[[3]]

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

obj <- readRDS(ref_rds)
assay <- obj@assays[[1]]

expr <- assay@data
if (!inherits(expr, "Matrix")) {
  expr <- as(as.matrix(expr), "dgCMatrix")
}

obs <- obj@meta.data
obs$cell_barcode <- rownames(obs)

dir.create(dirname(out_matrix), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(out_obs), recursive = TRUE, showWarnings = FALSE)

Matrix::writeMM(expr, out_matrix)
write.csv(obs, out_obs, row.names = FALSE)
write.csv(data.frame(gene = rownames(expr)), file = sub("\\.mtx$", "_genes.csv", out_matrix), row.names = FALSE)
write.csv(data.frame(cell_barcode = colnames(expr)), file = sub("\\.mtx$", "_barcodes.csv", out_matrix), row.names = FALSE)

cat("matrix=", out_matrix, "\n", sep = "")
cat("obs=", out_obs, "\n", sep = "")
