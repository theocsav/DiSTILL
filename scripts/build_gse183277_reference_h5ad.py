from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import io as spio


def load_metadata(metadata_gz: Path) -> pd.DataFrame:
    with gzip.open(metadata_gz, "rt", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
    columns = ["cell_barcode"] + header
    metadata = pd.read_csv(
        metadata_gz,
        sep="\t",
        compression="gzip",
        header=0,
        names=columns,
        low_memory=False,
    )
    metadata = metadata.set_index("cell_barcode", drop=False)
    return metadata


def build_reference_h5ad(
    matrix_mtx: Path,
    genes_csv: Path,
    barcodes_csv: Path,
    metadata_gz: Path,
    output_h5ad: Path,
    label_column: str,
) -> Path:
    matrix = spio.mmread(matrix_mtx).tocsr().transpose()
    genes = pd.read_csv(genes_csv)
    barcodes = pd.read_csv(barcodes_csv)
    metadata = load_metadata(metadata_gz)

    obs = metadata.loc[barcodes["cell_barcode"].tolist()].copy()
    obs["cell_type"] = obs[label_column].astype(str)
    var = pd.DataFrame(index=genes["gene"].astype(str))
    var.index.name = "gene"

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    return output_h5ad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a kidney reference.h5ad from GSE183277 counts and metadata.")
    parser.add_argument("--matrix-mtx", type=Path, required=True)
    parser.add_argument("--genes-csv", type=Path, required=True)
    parser.add_argument("--barcodes-csv", type=Path, required=True)
    parser.add_argument("--metadata-gz", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--label-column", type=str, default="subclass.l2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_reference_h5ad(
        matrix_mtx=args.matrix_mtx,
        genes_csv=args.genes_csv,
        barcodes_csv=args.barcodes_csv,
        metadata_gz=args.metadata_gz,
        output_h5ad=args.output_h5ad,
        label_column=args.label_column,
    )
    print(output)


if __name__ == "__main__":
    main()
