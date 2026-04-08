#!/usr/bin/env python3
import argparse
import os
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import BallTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


DEFAULT_NICHE_H5AD = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/Corrected_CompleteCosMx.h5ad"
DEFAULT_NEIGHBORHOOD_H5AD = (
    "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/CompleteCosMx_singlecellspatialresolution.h5ad"
)
DEFAULT_BASE_DIR = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis/RCausalMGM"
DEFAULT_NICHE_DIR = os.path.join(DEFAULT_BASE_DIR, "NicheCompositions")
DEFAULT_NEIGHBORHOOD_DIR = os.path.join(DEFAULT_BASE_DIR, "NeighborhoodInteractions")


def _coerce_field_of_view_label(value: object) -> str:
    return str(value).strip()


def _field_of_view_to_disease_state(value: object) -> str:
    text = _coerce_field_of_view_label(value)
    return text.rsplit("_", 1)[0] if "_" in text else text


def calculate_niche_compositions_percent(input_path: str, output_path: str) -> None:
    """Compute percent composition of NMF factors by field of view."""
    adata = ad.read_h5ad(input_path)
    if "NMF_factor" not in adata.obs.columns or "unique_cell_id" not in adata.obs.columns:
        raise RuntimeError("Required columns 'NMF_factor' or 'unique_cell_id' not found.")
    columns = ["unique_cell_id", "NMF_factor"]
    if "patient" in adata.obs.columns:
        columns.append("patient")
    if "fov" in adata.obs.columns:
        columns.append("fov")
    df = adata.obs[columns].copy()
    if "patient" in df.columns and "fov" in df.columns:
        df["field_of_view"] = df["patient"].astype(str) + "_" + df["fov"].astype(str)
    else:
        df["field_of_view"] = df["unique_cell_id"].astype(str).str.rsplit("_", n=1).str[0]
    niche_counts = pd.pivot_table(
        df,
        index="field_of_view",
        columns="NMF_factor",
        aggfunc="size",
        fill_value=0,
    )
    niche_percent = niche_counts.div(niche_counts.sum(axis=1), axis=0)
    niche_percent.to_csv(output_path)


def transform_and_rename(input_path: str, output_path: str) -> None:
    """Log-transform niche composition values and rename columns."""
    df = pd.read_csv(input_path)
    if "field_of_view" not in df.columns:
        raise RuntimeError("field_of_view column not found in niche composition output.")
    df = df.set_index("field_of_view")
    rename_dict = {col: f"Niche_{col}" for col in df.columns}
    df.rename(columns=rename_dict, inplace=True)
    df_transformed = -np.log(df + 1e-9)
    df_transformed.to_csv(output_path)


def add_disease_state(input_path: str, output_path: str) -> None:
    """Add Disease/Health State column based on field_of_view."""
    df = pd.read_csv(input_path)
    if "field_of_view" not in df.columns:
        raise RuntimeError("field_of_view column not found in niche composition output.")
    df["Disease/Health State"] = df["field_of_view"].apply(_field_of_view_to_disease_state)
    df.to_csv(output_path, index=False)


def compute_neighborhood_enrichment(input_path: str, output_path: str) -> None:
    """Compute neighborhood enrichment per FOV and save enrichment matrix."""
    adata = sc.read_h5ad(input_path)
    required_cols = {"patient", "fov", "CenterX_global_px", "CenterY_global_px", "NMF_factor", "Area"}
    missing = required_cols - set(adata.obs.columns)
    if missing:
        raise RuntimeError(f"Missing required obs columns: {', '.join(sorted(missing))}")
    adata.obs["Disease_State"] = adata.obs["patient"].astype(str).str[:2]
    adata.obs["unique_fov"] = adata.obs["patient"].astype(str) + "_" + adata.obs["fov"].astype(str)
    coords_um = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values.astype("float64")
    nmf_labels = adata.obs["NMF_factor"]
    cell_diameters_um = 2 * np.sqrt(adata.obs["Area"] / np.pi)
    all_factor_names = sorted(adata.obs["NMF_factor"].unique())
    obs_frame = adata.obs.copy()
    obs_frame["obs_name"] = adata.obs_names.astype(str)
    fov_groups = obs_frame.groupby("unique_fov")["obs_name"].apply(list)
    results = []
    for fov_id, cell_indices in fov_groups.items():
        original_indices = adata.obs.index.get_indexer(cell_indices)
        fov_coords = coords_um[original_indices]
        fov_cell_diameters = cell_diameters_um.loc[cell_indices].values
        fov_labels = nmf_labels.loc[cell_indices]
        if len(fov_coords) < 2:
            continue
        tree = BallTree(fov_coords)
        interaction_matrix = pd.DataFrame(0, index=all_factor_names, columns=all_factor_names)
        for i in range(len(fov_coords)):
            per_cell_threshold = 2 * fov_cell_diameters[i]
            neighbor_indices = tree.query_radius([fov_coords[i]], r=per_cell_threshold)[0]
            neighbor_indices = neighbor_indices[neighbor_indices != i]
            if len(neighbor_indices) == 0:
                continue
            factor_i = fov_labels.iloc[i]
            factors_j = fov_labels.iloc[neighbor_indices]
            counts = factors_j.value_counts()
            for factor, count in counts.items():
                interaction_matrix.loc[factor_i, factor] += count
        interaction_matrix = interaction_matrix + interaction_matrix.T
        niche_proportions = fov_labels.value_counts(normalize=True).reindex(all_factor_names, fill_value=0)
        total_interactions = interaction_matrix.sum().sum()
        expected_matrix = total_interactions * np.outer(niche_proportions, niche_proportions)
        expected_matrix = pd.DataFrame(expected_matrix, index=all_factor_names, columns=all_factor_names)
        enrichment = np.log2((interaction_matrix + 1) / (expected_matrix + 1))
        row = {"field_of_view": fov_id}
        for i, fi in enumerate(all_factor_names, 1):
            for j, fj in enumerate(all_factor_names, 1):
                row[f"enrichment_{i}-{j}"] = enrichment.loc[fi, fj]
        results.append(row)
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)


def add_neighborhood_disease_state(input_path: str, output_path: str) -> None:
    """Add Disease/Health State column for neighborhood enrichment outputs."""
    df = pd.read_csv(input_path)
    df["Disease/Health State"] = df["field_of_view"].apply(_field_of_view_to_disease_state)
    df.to_csv(output_path, index=False)


def write_correlation_outputs(input_path: str) -> tuple[str, str]:
    """Write correlation CSV and heatmap PNG for enrichment columns."""
    df = pd.read_csv(input_path)
    enrichment_cols = [col for col in df.columns if col.startswith("enrichment_")]
    enrichment_data = df[enrichment_cols]
    cor_matrix = enrichment_data.corr()
    csv_path = input_path.replace(".csv", "_correlation_matrix.csv")
    cor_matrix.to_csv(csv_path)
    heatmap_path = input_path.replace(".csv", "_correlation_heatmap.png")
    plt.figure(figsize=(12, 10))
    sns.heatmap(cor_matrix, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Matrix of Enrichment Scores")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    return csv_path, heatmap_path


def write_high_res_heatmap(input_path: str, output_path: str) -> None:
    """Write a high resolution correlation heatmap."""
    df = pd.read_csv(input_path)
    enrichment_cols = [col for col in df.columns if col.startswith("enrichment_")]
    enrichment_data = df[enrichment_cols]
    cor_matrix = enrichment_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(cor_matrix, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Matrix of Enrichment Scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def drop_collinear_columns(input_path: str, output_path: str) -> None:
    """Drop symmetrical enrichment columns and save reduced output."""
    df = pd.read_csv(input_path)
    enrichment_cols = [col for col in df.columns if col.startswith("enrichment_")]
    to_drop = set()
    seen_pairs = set()
    for col in enrichment_cols:
        match = re.match(r"enrichment_(\d+)-(\d+)", col)
        if match:
            a, b = match.groups()
            pair = tuple(sorted([a, b]))
            if pair in seen_pairs:
                to_drop.add(col)
            else:
                seen_pairs.add(pair)
    df_reduced = df.drop(columns=list(to_drop))
    df_reduced.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="RCausalMGM preparation steps.")
    parser.add_argument("--output-dir", default=None, help="Base output directory for RCausalMGM artifacts.")
    parser.add_argument(
        "--niche-output-dir",
        default=None,
        help="Override output directory for niche composition outputs.",
    )
    parser.add_argument(
        "--neighborhood-output-dir",
        default=None,
        help="Override output directory for neighborhood interaction outputs.",
    )
    parser.add_argument(
        "--niche-h5ad",
        default=DEFAULT_NICHE_H5AD,
        help="Input h5ad for niche composition calculations.",
    )
    parser.add_argument(
        "--neighborhood-h5ad",
        default=DEFAULT_NEIGHBORHOOD_H5AD,
        help="Input h5ad for neighborhood interaction calculations.",
    )
    args = parser.parse_args()

    base_dir = args.output_dir or DEFAULT_BASE_DIR
    niche_output_dir = args.niche_output_dir or os.path.join(base_dir, "NicheCompositions")
    neighborhood_output_dir = args.neighborhood_output_dir or os.path.join(base_dir, "NeighborhoodInteractions")
    os.makedirs(niche_output_dir, exist_ok=True)
    os.makedirs(neighborhood_output_dir, exist_ok=True)

    niche_percent_path = os.path.join(niche_output_dir, "niche_compositions_percent.csv")
    niche_log_path = os.path.join(niche_output_dir, "niche_compositions_log_transformed.csv")
    niche_final_path = os.path.join(niche_output_dir, "niche_compositions_final.csv")

    calculate_niche_compositions_percent(args.niche_h5ad, niche_percent_path)
    transform_and_rename(niche_percent_path, niche_log_path)
    add_disease_state(niche_log_path, niche_final_path)

    neighborhood_enrichment_path = os.path.join(neighborhood_output_dir, "FOV_Neighborhood_Enrichment.csv")
    compute_neighborhood_enrichment(args.neighborhood_h5ad, neighborhood_enrichment_path)

    neighborhood_with_disease = neighborhood_enrichment_path.replace(".csv", "_withDisease.csv")
    add_neighborhood_disease_state(neighborhood_enrichment_path, neighborhood_with_disease)

    write_correlation_outputs(neighborhood_with_disease)
    high_res_path = os.path.join(neighborhood_output_dir, "correlation_matrix.png")
    write_high_res_heatmap(neighborhood_with_disease, high_res_path)

    no_collinear_path = neighborhood_with_disease.replace(".csv", "_noCollinear.csv")
    drop_collinear_columns(neighborhood_with_disease, no_collinear_path)


if __name__ == "__main__":
    main()
