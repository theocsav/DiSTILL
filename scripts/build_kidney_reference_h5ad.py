from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import io as spio


def build_reference_h5ad(matrix_mtx: Path, obs_csv: Path, output_h5ad: Path, label_column: str) -> Path:
    matrix = spio.mmread(matrix_mtx).tocsr().transpose()
    obs = pd.read_csv(obs_csv)
    genes = pd.read_csv(matrix_mtx.with_name(matrix_mtx.stem + "_genes.csv"))
    barcodes = pd.read_csv(matrix_mtx.with_name(matrix_mtx.stem + "_barcodes.csv"))

    obs = obs.set_index("cell_barcode")
    obs = obs.loc[barcodes["cell_barcode"].tolist()]
    obs["cell_type"] = obs[label_column].astype(str)

    var = pd.DataFrame(index=genes["gene"].astype(str))
    var.index.name = "gene"

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    return output_h5ad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a kidney reference.h5ad from an exported Seurat RDS matrix.")
    parser.add_argument("--matrix-mtx", type=Path, required=True)
    parser.add_argument("--obs-csv", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--label-column", type=str, default="annotation.l2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_reference_h5ad(
        matrix_mtx=args.matrix_mtx,
        obs_csv=args.obs_csv,
        output_h5ad=args.output_h5ad,
        label_column=args.label_column,
    )
    print(output)


if __name__ == "__main__":
    main()
