from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


VISIUM_SPOT_DIAMETER_UM = 55.0


def _require_columns(obs: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise RuntimeError(f"Source h5ad is missing required obs columns: {missing}")


def _retile_obs(obs: pd.DataFrame, pseudo_fov_tile_um: float) -> pd.DataFrame:
    obs = obs.copy()
    _require_columns(
        obs,
        [
            "patient",
            "Disease_State",
            "CenterX_global_px",
            "CenterY_global_px",
            "Width",
            "unique_cell_id",
        ],
    )

    width_px = pd.to_numeric(obs["Width"], errors="coerce")
    if width_px.isna().all():
        raise RuntimeError("Width column could not be converted to numeric values.")
    px_per_um = float(width_px.median()) / VISIUM_SPOT_DIAMETER_UM
    tile_px = float(pseudo_fov_tile_um) * px_per_um

    retiled_frames: list[pd.DataFrame] = []
    for patient, sample_df in obs.groupby("patient", observed=False):
        sample_df = sample_df.copy()
        x0 = float(sample_df["CenterX_global_px"].min())
        y0 = float(sample_df["CenterY_global_px"].min())
        sample_df["pseudo_fov_tile_um"] = float(pseudo_fov_tile_um)
        sample_df["pseudo_fov_tile_px"] = float(tile_px)
        sample_df["tile_x"] = np.floor((sample_df["CenterX_global_px"].to_numpy() - x0) / tile_px).astype(int)
        sample_df["tile_y"] = np.floor((sample_df["CenterY_global_px"].to_numpy() - y0) / tile_px).astype(int)
        sample_df["fov"] = (
            "tile"
            + sample_df["tile_x"].astype(str)
            + "_"
            + sample_df["tile_y"].astype(str)
        )
        sample_df["field_of_view"] = sample_df["patient"].astype(str) + "_" + sample_df["fov"].astype(str)
        sample_df["unique_fov"] = sample_df["field_of_view"]
        retiled_frames.append(sample_df)

    retiled = pd.concat(retiled_frames, axis=0)
    retiled["patient"] = retiled["patient"].astype("category")
    retiled["Disease_State"] = retiled["Disease_State"].astype("category")
    retiled["fov"] = retiled["fov"].astype("category")
    return retiled


def build_retiled_h5ad(
    source_h5ad: Path,
    output_dir: Path,
    dataset_id: str,
    pseudo_fov_tile_um: float,
) -> tuple[Path, Path, Path]:
    adata = ad.read_h5ad(source_h5ad)
    retiled_obs = _retile_obs(adata.obs, pseudo_fov_tile_um=pseudo_fov_tile_um)

    adata = adata[retiled_obs.index].copy()
    adata.obs = retiled_obs
    adata.obs_names = adata.obs["unique_cell_id"].astype(str)
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(dtype="float64")

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_h5ad_path = output_dir / f"{dataset_id}_spatial.h5ad"
    metadata_csv_path = output_dir / f"{dataset_id}_metadata.csv"
    manifest_csv_path = output_dir / f"{dataset_id}_sample_manifest.csv"

    manifest = (
        adata.obs.groupby("patient", observed=False)
        .agg(
            disease_state=("Disease_State", "first"),
            n_spots_in_tissue=("unique_cell_id", "size"),
            pseudo_fov_count=("field_of_view", "nunique"),
        )
        .reset_index()
        .rename(columns={"patient": "sample_id"})
    )
    manifest["source_h5ad"] = str(source_h5ad)
    manifest["pseudo_fov_tile_um"] = float(pseudo_fov_tile_um)

    adata.write_h5ad(spatial_h5ad_path)
    adata.obs.reset_index(drop=True).to_csv(metadata_csv_path, index=False)
    manifest.to_csv(manifest_csv_path, index=False)
    return spatial_h5ad_path, metadata_csv_path, manifest_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retile an existing processed skin Visium h5ad into new pseudo-FOV sizes.")
    parser.add_argument("--source-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--pseudo-fov-tile-um", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spatial_h5ad_path, metadata_csv_path, manifest_csv_path = build_retiled_h5ad(
        source_h5ad=args.source_h5ad,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        pseudo_fov_tile_um=args.pseudo_fov_tile_um,
    )
    print(f"spatial_h5ad={spatial_h5ad_path}")
    print(f"cell_metadata={metadata_csv_path}")
    print(f"sample_manifest={manifest_csv_path}")


if __name__ == "__main__":
    main()
