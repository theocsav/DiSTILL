#!/usr/bin/env python3
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _pick_column(columns, candidates, default=None):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return default


def _sorted_factor_labels(values):
    labels = sorted({str(v) for v in values})
    return sorted(labels, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def _build_obs_frame(h5ad_path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs.copy()
    if "patient" not in obs.columns:
        raise RuntimeError("cosmx_with_nmf.h5ad is missing 'patient' in .obs.")

    disease_col = _pick_column(list(obs.columns), ["disease_state", "Disease_State", "Disease/Health State"])
    if disease_col is None:
        raise RuntimeError("cosmx_with_nmf.h5ad is missing a disease-state column.")

    nmf_col = _pick_column(list(obs.columns), ["NMF_factor", "dominant_nmf_factor"])
    if nmf_col is None:
        raise RuntimeError("cosmx_with_nmf.h5ad is missing NMF_factor/dominant_nmf_factor.")

    obs = obs.copy()
    obs["patient"] = obs["patient"].astype(str)
    obs["disease_state"] = obs[disease_col].astype(str)
    obs["nmf_factor"] = obs[nmf_col].astype(str)

    if "unique_fov" in obs.columns:
        obs["fov_key"] = obs["unique_fov"].astype(str)
    elif "fov" in obs.columns:
        obs["fov_key"] = obs["patient"].astype(str) + "_" + obs["fov"].astype(str)
    else:
        raise RuntimeError("cosmx_with_nmf.h5ad is missing 'fov' or 'unique_fov' needed for FOV-level aggregation.")

    return obs


def _read_feature_frame(path_no_ext: Path) -> pd.DataFrame:
    parquet_path = path_no_ext.with_suffix(".parquet")
    csv_path = path_no_ext.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FOV-level MLP inputs from Post-NMF artifacts.")
    parser.add_argument("--output-dir", required=True, help="Run outputs directory containing Post-NMF artifacts.")
    parser.add_argument("--cosmx-with-nmf", required=True, help="Path to cosmx_with_nmf.h5ad.")
    parser.add_argument("--dest-dir", required=True, help="Destination directory for MLP input parquet files.")
    parser.add_argument("--niche-gene-count", type=int, default=20, help="Top ranked niche-gene features to keep per group.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    obs = _build_obs_frame(Path(args.cosmx_with_nmf))
    fov_index = pd.Index(sorted(obs["fov_key"].astype(str).unique()), name="fov_key")

    fov_nmf_counts = pd.crosstab(obs["fov_key"], obs["nmf_factor"]).reindex(fov_index, fill_value=0)
    fov_nmf_props = fov_nmf_counts.div(fov_nmf_counts.sum(axis=1), axis=0).fillna(0.0)
    fov_nmf_props.columns = [f"nmf_prop_{col}" for col in fov_nmf_props.columns]

    enrichment_fov = _read_feature_frame(output_dir / "enrichment_features_fov")
    enrichment_fov.index = enrichment_fov.index.astype(str)
    enrichment_fov = enrichment_fov.reindex(fov_index).fillna(0.0)
    full_enrichment_cols = []
    for col in enrichment_fov.columns:
        if not str(col).startswith("enrichment_"):
            continue
        try:
            left, right = str(col).replace("enrichment_", "").split("-")
            if int(left) < int(right):
                full_enrichment_cols.append(col)
        except ValueError:
            continue
    enrichment_fov = enrichment_fov[full_enrichment_cols]

    niche_gene_fov = _read_feature_frame(output_dir / "niche_gene_features_fov")
    niche_gene_fov.index = niche_gene_fov.index.astype(str)
    niche_gene_fov = niche_gene_fov.reindex(fov_index).fillna(0.0)

    ranked_path = output_dir / "niche_gene_ranked_features.csv"
    selected_niche_gene = []
    if ranked_path.exists():
        ranked = pd.read_csv(ranked_path)
        if {"group", "feature"}.issubset(ranked.columns):
            for group_name, sub in ranked.groupby("group", sort=False):
                for feature in sub["feature"].astype(str).head(max(0, args.niche_gene_count)):
                    if feature in niche_gene_fov.columns and feature not in selected_niche_gene:
                        selected_niche_gene.append(feature)

    combined = (
        fov_nmf_props
        .join(enrichment_fov, how="left")
        .join(niche_gene_fov[selected_niche_gene], how="left")
        .fillna(0.0)
    )
    combined.index.name = "fov_key"

    fov_disease = obs.groupby("fov_key")["disease_state"].first().reindex(combined.index)
    fov_patient = obs.groupby("fov_key")["patient"].first().reindex(combined.index)

    combined.to_parquet(dest_dir / "combined_features_filtered.parquet")
    combined.to_csv(dest_dir / "combined_features_filtered.csv")
    fov_disease.to_frame("Disease_State").to_parquet(dest_dir / "targets_y.parquet")
    fov_patient.to_frame("patient").to_parquet(dest_dir / "groups.parquet")
    pd.DataFrame(
        {
            "fov_key": combined.index.astype(str),
            "patient": fov_patient.astype(str).values,
            "Disease_State": fov_disease.astype(str).values,
        }
    ).to_csv(dest_dir / "fov_metadata.csv", index=False)

    print(f"Wrote FOV-level combined features to {dest_dir}")
    print(f"Rows: {combined.shape[0]}, columns: {combined.shape[1]}")


if __name__ == "__main__":
    main()
