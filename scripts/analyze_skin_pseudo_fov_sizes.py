#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


VISIUM_SPOT_DIAMETER_UM = 55.0


def _tile_sample(sample_df: pd.DataFrame, tile_um: float) -> pd.DataFrame:
    width_px = float(sample_df["Width"].median())
    px_per_um = width_px / VISIUM_SPOT_DIAMETER_UM
    tile_px = tile_um * px_per_um

    x0 = float(sample_df["CenterX_global_px"].min())
    y0 = float(sample_df["CenterY_global_px"].min())

    tile_x = np.floor((sample_df["CenterX_global_px"].to_numpy() - x0) / tile_px).astype(int)
    tile_y = np.floor((sample_df["CenterY_global_px"].to_numpy() - y0) / tile_px).astype(int)

    tiled = sample_df.copy()
    tiled["tile_um"] = int(tile_um)
    tiled["tile_px"] = tile_px
    tiled["tile_x"] = tile_x
    tiled["tile_y"] = tile_y
    tiled["pseudo_fov_id"] = (
        tiled["patient"].astype(str)
        + f"_tile{int(tile_um)}um_"
        + tiled["tile_x"].astype(str)
        + "_"
        + tiled["tile_y"].astype(str)
    )
    return tiled


def _summarize_tiles(tiled: pd.DataFrame, min_spots: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tile_counts = (
        tiled.groupby(["patient", "Disease_State", "tile_um", "pseudo_fov_id", "tile_x", "tile_y"], observed=False)
        .size()
        .reset_index(name="spots_in_tile")
    )
    tile_counts["meets_min_spots"] = tile_counts["spots_in_tile"] >= min_spots

    summary = (
        tile_counts.groupby(["patient", "Disease_State", "tile_um"], observed=False)
        .agg(
            pseudo_fovs_total=("pseudo_fov_id", "nunique"),
            pseudo_fovs_ge_min=("meets_min_spots", "sum"),
            mean_spots_per_fov=("spots_in_tile", "mean"),
            median_spots_per_fov=("spots_in_tile", "median"),
            max_spots_per_fov=("spots_in_tile", "max"),
        )
        .reset_index()
    )
    summary["fraction_ge_min"] = summary["pseudo_fovs_ge_min"] / summary["pseudo_fovs_total"]
    return tile_counts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pseudo-FOV tile sizes for skin Visium samples.")
    parser.add_argument("--h5ad", required=True, help="Path to processed spatial h5ad.")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis CSVs.")
    parser.add_argument(
        "--tile-sizes-um",
        nargs="+",
        type=float,
        default=[500.0, 1000.0, 2000.0],
        help="Pseudo-FOV tile sizes in microns.",
    )
    parser.add_argument("--min-spots", type=int, default=50, help="Minimum acceptable spots per pseudo-FOV.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    required = {"patient", "Disease_State", "CenterX_global_px", "CenterY_global_px", "Width"}
    missing = sorted(required - set(adata.obs.columns))
    if missing:
        raise RuntimeError(f"Missing required obs columns: {', '.join(missing)}")

    obs = adata.obs[list(required)].copy()
    obs["patient"] = obs["patient"].astype(str)
    obs["Disease_State"] = obs["Disease_State"].astype(str)

    all_tile_counts: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    for tile_um in args.tile_sizes_um:
        tiled_frames = []
        for _, sample_df in obs.groupby("patient", observed=False):
            tiled_frames.append(_tile_sample(sample_df, tile_um))
        tiled = pd.concat(tiled_frames, ignore_index=True)
        tile_counts, summary = _summarize_tiles(tiled, args.min_spots)
        all_tile_counts.append(tile_counts)
        all_summary.append(summary)

    tile_counts_df = pd.concat(all_tile_counts, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    cohort_summary = (
        summary_df.groupby(["Disease_State", "tile_um"], observed=False)
        .agg(
            samples=("patient", "nunique"),
            pseudo_fovs_total=("pseudo_fovs_total", "sum"),
            pseudo_fovs_ge_min=("pseudo_fovs_ge_min", "sum"),
            mean_pseudo_fovs_per_sample=("pseudo_fovs_total", "mean"),
            mean_valid_pseudo_fovs_per_sample=("pseudo_fovs_ge_min", "mean"),
            median_spots_per_fov=("median_spots_per_fov", "median"),
            max_spots_per_fov=("max_spots_per_fov", "max"),
        )
        .reset_index()
    )
    cohort_summary["fraction_ge_min"] = (
        cohort_summary["pseudo_fovs_ge_min"] / cohort_summary["pseudo_fovs_total"]
    )

    tile_counts_path = output_dir / "skin_visium_pseudo_fov_tile_counts.csv"
    summary_path = output_dir / "skin_visium_pseudo_fov_summary_by_sample.csv"
    cohort_path = output_dir / "skin_visium_pseudo_fov_summary_by_cohort.csv"
    tile_counts_df.to_csv(tile_counts_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    cohort_summary.to_csv(cohort_path, index=False)

    print(f"Wrote {tile_counts_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {cohort_path}")
    print("\nPer-sample summary:")
    print(summary_df.to_string(index=False))
    print("\nCohort summary:")
    print(cohort_summary.to_string(index=False))


if __name__ == "__main__":
    main()
