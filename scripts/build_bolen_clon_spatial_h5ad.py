from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse


DEFAULT_SAMPLE_ID = "Bolen_CLON39_R1000"


def load_inputs(expr_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr = pd.read_csv(expr_path)
    metadata = pd.read_csv(metadata_path)
    return expr, metadata


def build_metadata_frame(metadata: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    obs = metadata.copy()
    obs["patient"] = sample_id
    obs["cell_ID"] = obs["cell_ID"].astype(str)
    obs["cell_id"] = obs["cell_id"].astype(str)
    obs["cell"] = obs["cell"].astype(str)
    obs["fov"] = obs["fov"].astype(str)
    obs["unique_cell_id"] = sample_id + "_" + obs["fov"] + "_" + obs["cell_ID"]
    obs = obs.set_index("unique_cell_id", drop=False)
    return obs


def build_expression_matrix(
    expr: pd.DataFrame, obs: pd.DataFrame, sample_id: str
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    expr_key = expr[["fov", "cell_ID", "cell"]].copy()
    expr_key["fov"] = expr_key["fov"].astype(str)
    expr_key["cell_ID"] = expr_key["cell_ID"].astype(str)
    expr_key["cell"] = expr_key["cell"].astype(str)
    expr_key["unique_cell_id"] = (
        sample_id + "_" + expr_key["fov"] + "_" + expr_key["cell_ID"]
    )
    expr = expr.copy()
    expr["unique_cell_id"] = expr_key["unique_cell_id"]
    expr = expr.set_index("unique_cell_id", drop=True)

    gene_columns = [column for column in expr.columns if column not in {"fov", "cell_ID", "cell"}]
    expr = expr.loc[obs.index]
    matrix = sparse.csr_matrix(expr[gene_columns].to_numpy(dtype="int32"))
    var = pd.DataFrame(index=gene_columns)
    var.index.name = "gene"
    return matrix, var


def build_spatial_h5ad(expr_path: Path, metadata_path: Path, output_dir: Path, sample_id: str) -> tuple[Path, Path]:
    expr, metadata = load_inputs(expr_path, metadata_path)
    obs = build_metadata_frame(metadata, sample_id)
    matrix, var = build_expression_matrix(expr, obs, sample_id)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obs["patient"] = adata.obs["patient"].astype("category")
    adata.obs["fov"] = adata.obs["fov"].astype("category")
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(dtype="float64")

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_h5ad_path = output_dir / f"{sample_id}_spatial.h5ad"
    metadata_csv_path = output_dir / f"{sample_id}_cell_metadata.csv"

    adata.write_h5ad(spatial_h5ad_path)
    obs.to_csv(metadata_csv_path, index=False)
    return spatial_h5ad_path, metadata_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a spatial h5ad from Bolen CLON CosMx CSV files.")
    parser.add_argument(
        "--expr",
        type=Path,
        default=Path("bolen_clon") / "02_Bolen_CLON39_R1000_exprMat_file.csv",
        help="Path to the CosMx expression matrix CSV.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("bolen_clon") / "02_Bolen_CLON39_R1000_metadata_file.csv",
        help="Path to the CosMx metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bolen_clon") / "processed",
        help="Directory where the generated files will be written.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=DEFAULT_SAMPLE_ID,
        help="Sample identifier used to populate the patient column and unique cell ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spatial_h5ad_path, metadata_csv_path = build_spatial_h5ad(
        expr_path=args.expr,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
    )
    print(f"spatial_h5ad={spatial_h5ad_path}")
    print(f"cell_metadata={metadata_csv_path}")
    print("reference_h5ad=missing")


if __name__ == "__main__":
    main()
