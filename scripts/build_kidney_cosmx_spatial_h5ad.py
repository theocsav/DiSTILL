from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse


OUTER_ZIP_DEFAULT = Path("kidney_dataset") / "41467_2025_60034_MOESM4_ESM.zip"
DEFAULT_SAMPLES = ("HD1", "HD2", "HD3", "SSc1", "SSc2", "SSc3")


def _read_nested_csv(outer_zip: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(outer_zip) as outer:
        payload = outer.read(member_name)
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        inner_name = inner.namelist()[0]
        with inner.open(inner_name) as handle:
            return pd.read_csv(handle)


def _load_sample_frames(outer_zip: Path, sample_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr_member = f"Source Data/Kidney_Spatial_transcriptome_{sample_id}_exprMat_data.zip"
    metadata_member = f"Source Data/Kidney_Spatial_transcriptome_{sample_id}_metadata_data.zip"
    expr = _read_nested_csv(outer_zip, expr_member)
    metadata = _read_nested_csv(outer_zip, metadata_member)
    return expr, metadata


def _normalize_metadata(metadata: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    obs = metadata.copy()
    obs["patient"] = sample_id
    obs["Disease_State"] = "healthy" if sample_id.startswith("HD") else "systemic_sclerosis"
    obs["fov"] = obs["fov"].astype(str)
    obs["cell_ID"] = obs["cell_ID"].astype(str)
    obs["cell_id"] = obs["cell_id"].astype(str)
    obs["cell"] = obs["cell"].astype(str)
    obs["unique_cell_id"] = sample_id + "_" + obs["fov"] + "_" + obs["cell_ID"]
    obs = obs.set_index("unique_cell_id", drop=False)
    return obs


def _build_expression_matrix(
    expr: pd.DataFrame,
    obs: pd.DataFrame,
    sample_id: str,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    expr = expr.copy()
    expr["fov"] = expr["fov"].astype(str)
    expr["cell_ID"] = expr["cell_ID"].astype(str)
    expr["unique_cell_id"] = sample_id + "_" + expr["fov"] + "_" + expr["cell_ID"]
    expr = expr.set_index("unique_cell_id", drop=True)

    drop_columns = {"Unnamed: 0", "fov", "cell_ID"}
    gene_columns = [column for column in expr.columns if column not in drop_columns]
    expr = expr.loc[obs.index]
    matrix = sparse.csr_matrix(expr[gene_columns].to_numpy(dtype="int32"))
    var = pd.DataFrame(index=gene_columns)
    var.index.name = "gene"
    return matrix, var


def build_spatial_h5ad(
    outer_zip: Path,
    output_dir: Path,
    sample_ids: list[str],
    dataset_id: str,
) -> tuple[Path, Path]:
    adatas: list[ad.AnnData] = []
    metadata_frames: list[pd.DataFrame] = []

    for sample_id in sample_ids:
        expr, metadata = _load_sample_frames(outer_zip, sample_id)
        obs = _normalize_metadata(metadata, sample_id)
        matrix, var = _build_expression_matrix(expr, obs, sample_id)
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
        adata.obs["patient"] = adata.obs["patient"].astype("category")
        adata.obs["Disease_State"] = adata.obs["Disease_State"].astype("category")
        adata.obs["fov"] = adata.obs["fov"].astype("category")
        adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(dtype="float64")
        adatas.append(adata)
        metadata_frames.append(obs.reset_index(drop=True))

    combined = ad.concat(adatas, axis=0, merge="same", join="inner", index_unique=None)
    combined.obs_names = combined.obs["unique_cell_id"].astype(str)

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_h5ad_path = output_dir / f"{dataset_id}_spatial.h5ad"
    metadata_csv_path = output_dir / f"{dataset_id}_cell_metadata.csv"

    combined.write_h5ad(spatial_h5ad_path)
    pd.concat(metadata_frames, ignore_index=True).to_csv(metadata_csv_path, index=False)
    return spatial_h5ad_path, metadata_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined kidney CosMX spatial h5ad from the paper source-data zip.")
    parser.add_argument(
        "--outer-zip",
        type=Path,
        default=OUTER_ZIP_DEFAULT,
        help="Path to the MOESM source-data zip that contains the per-sample CosMX archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kidney_dataset") / "processed",
        help="Directory where the generated files will be written.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="kidney_cosmx_six_sample",
        help="Prefix used for the generated output filenames.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(DEFAULT_SAMPLES),
        help="Sample IDs to include from the source-data zip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spatial_h5ad_path, metadata_csv_path = build_spatial_h5ad(
        outer_zip=args.outer_zip,
        output_dir=args.output_dir,
        sample_ids=args.samples,
        dataset_id=args.dataset_id,
    )
    print(f"spatial_h5ad={spatial_h5ad_path}")
    print(f"cell_metadata={metadata_csv_path}")
    print("reference_h5ad=missing_cell_type_annotations")


if __name__ == "__main__":
    main()
