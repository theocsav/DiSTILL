#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _first_non_null(values: pd.Series, fallback: str) -> str:
    values = values.dropna().astype(str)
    return values.iloc[0] if not values.empty else fallback


def _load_ranked_niche_gene_features(
    ranked_path: Path,
    default_group_count: int,
) -> tuple[dict[str, list[str]], list[str]]:
    if not ranked_path.exists():
        return {}, []
    ranked = pd.read_csv(ranked_path)
    if ranked.empty or "group" not in ranked.columns or "feature" not in ranked.columns:
        return {}, []
    rankings: dict[str, list[str]] = {}
    for group, group_df in ranked.groupby("group", sort=False):
        features = group_df.sort_values("rank")["feature"].astype(str).tolist()
        rankings[str(group)] = features

    selected: list[str] = []
    if len(rankings) == 2:
        for group in rankings:
            for feature in rankings[group][:default_group_count]:
                if feature not in selected:
                    selected.append(feature)
    else:
        for group in rankings:
            for feature in rankings[group][:default_group_count]:
                if feature not in selected:
                    selected.append(feature)
    return rankings, selected


def build_inputs(output_dir: Path, default_group_count: int = 20) -> None:
    obs = pd.read_csv(output_dir / "post_nmf_obs.csv")
    required_obs = {"patient", "disease_state", "fov", "NMF_factor"}
    missing_obs = required_obs - set(obs.columns)
    if missing_obs:
        raise RuntimeError(f"Missing required columns in post_nmf_obs.csv: {sorted(missing_obs)}")

    obs["patient"] = obs["patient"].astype(str)
    obs["disease_state"] = obs["disease_state"].astype(str)
    obs["fov"] = obs["fov"].astype(str)
    obs["fov_key"] = obs["patient"] + "_" + obs["fov"]
    obs["NMF_factor"] = obs["NMF_factor"].astype(str)

    fov_disease = obs.groupby("fov_key")["disease_state"].agg(lambda s: _first_non_null(s, "unknown"))
    fov_patient = obs.groupby("fov_key")["patient"].agg(lambda s: _first_non_null(s, "unknown_patient"))

    fov_nmf_counts = pd.crosstab(obs["fov_key"], obs["NMF_factor"]).sort_index()
    fov_nmf_props = fov_nmf_counts.div(fov_nmf_counts.sum(axis=1), axis=0).fillna(0.0)
    fov_nmf_props.columns = [f"nmf_prop_{col}" for col in fov_nmf_props.columns]
    fov_nmf_props.index.name = "field_of_view"

    enrichment_fov = pd.read_csv(output_dir / "enrichment_features_fov.csv", index_col=0)
    enrichment_fov.index = enrichment_fov.index.astype(str)
    enrichment_feature_cols = [c for c in enrichment_fov.columns if c.startswith("enrichment_")]
    full_enrichment_feature_cols: list[str] = []
    for col in enrichment_feature_cols:
        try:
            left, right = col.replace("enrichment_", "").split("-")
            if int(left) < int(right):
                full_enrichment_feature_cols.append(col)
        except Exception:
            continue
    enrichment_features = enrichment_fov[full_enrichment_feature_cols].copy() if full_enrichment_feature_cols else pd.DataFrame(index=enrichment_fov.index)
    enrichment_features.index.name = "field_of_view"

    niche_gene_fov = pd.read_csv(output_dir / "niche_gene_features_fov.csv", index_col=0)
    niche_gene_fov.index = niche_gene_fov.index.astype(str)
    ranked_path = output_dir / "niche_gene_ranked_features.csv"
    rankings, selected_niche_gene = _load_ranked_niche_gene_features(ranked_path, default_group_count)
    if selected_niche_gene:
        selected_niche_gene = [c for c in selected_niche_gene if c in niche_gene_fov.columns]
        niche_gene_selected = niche_gene_fov[selected_niche_gene].copy()
    else:
        niche_gene_selected = pd.DataFrame(index=niche_gene_fov.index)
    niche_gene_selected.index.name = "field_of_view"

    common_index = (
        fov_nmf_props.index
        .intersection(enrichment_features.index)
        .intersection(niche_gene_fov.index)
    )
    combined = (
        fov_nmf_props.reindex(common_index)
        .join(enrichment_features.reindex(common_index), how="left")
        .join(niche_gene_selected.reindex(common_index), how="left")
        .fillna(0.0)
    )
    combined.index.name = "field_of_view"
    combined.to_parquet(output_dir / "combined_features_filtered.parquet")
    combined.to_csv(output_dir / "combined_features_filtered.csv")

    targets = fov_disease.reindex(common_index).to_frame("Disease_State")
    targets.index.name = "field_of_view"
    targets.to_parquet(output_dir / "targets_y.parquet")

    groups = fov_patient.reindex(common_index).to_frame("patient")
    groups.index.name = "field_of_view"
    groups.to_parquet(output_dir / "groups.parquet")

    priority_columns = [c for c in fov_nmf_props.columns if c in combined.columns]
    selected_enrichment_features: list[str] = []
    enrichment_sig_path = output_dir / "enrichment_significant_features.csv"
    if enrichment_sig_path.exists():
        enrichment_sig = pd.read_csv(enrichment_sig_path)
        if "feature" in enrichment_sig.columns:
            selected_enrichment_features = [
                c for c in enrichment_sig["feature"].astype(str).tolist() if c in combined.columns
            ]
    selected_niche_gene_in_combined = [c for c in niche_gene_selected.columns if c in combined.columns]
    remaining_slots = max(0, min(15, len(combined.columns)) - len(priority_columns))
    if remaining_slots > 0:
        enrichment_quota = min(len(selected_enrichment_features), remaining_slots // 2 if selected_niche_gene_in_combined else remaining_slots)
        niche_quota = min(len(selected_niche_gene_in_combined), remaining_slots - enrichment_quota)
        priority_columns.extend([c for c in selected_enrichment_features if c not in priority_columns][:enrichment_quota])
        priority_columns.extend([c for c in selected_niche_gene_in_combined if c not in priority_columns][:niche_quota])
        if len(priority_columns) < min(15, len(combined.columns)):
            remaining = [c for c in combined.columns if c not in priority_columns]
            remaining = sorted(remaining, key=lambda c: float(combined[c].var()), reverse=True)
            priority_columns.extend(remaining)
    reduced = combined[priority_columns[: min(15, len(priority_columns))]].copy()
    reduced.to_parquet(output_dir / "reduced_features_final_15.parquet")
    reduced.to_csv(output_dir / "reduced_features_final_15.csv")

    feature_sweep_dir = output_dir / "feature_sweeps"
    feature_sweep_dir.mkdir(exist_ok=True)
    base_feature_columns = [c for c in list(fov_nmf_props.columns) + list(enrichment_features.columns) if c in combined.columns]
    manifest = []
    if rankings:
        for niche_gene_count in [0, 5, 10, 15, 20]:
            selected: list[str] = []
            if len(rankings) == 2:
                for group in rankings:
                    for feature in rankings[group][:niche_gene_count]:
                        if feature in niche_gene_fov.columns and feature not in selected:
                            selected.append(feature)
            else:
                for group in rankings:
                    for feature in rankings[group][:niche_gene_count]:
                        if feature in niche_gene_fov.columns and feature not in selected:
                            selected.append(feature)
            sweep = (
                fov_nmf_props.reindex(common_index)
                .join(enrichment_features.reindex(common_index), how="left")
                .join(niche_gene_fov.reindex(common_index)[selected], how="left")
                .fillna(0.0)
            )
            sweep.index.name = "field_of_view"
            stem = f"combined_features_niche_gene_top_{niche_gene_count}_per_group"
            sweep.to_parquet(feature_sweep_dir / f"{stem}.parquet")
            sweep.to_csv(feature_sweep_dir / f"{stem}.csv")
            manifest.append(
                {
                    "niche_gene_count_per_group": niche_gene_count,
                    "feature_count": int(sweep.shape[1]),
                    "niche_gene_feature_count_after_overlap_removal": len(selected),
                    "parquet": str(feature_sweep_dir / f"{stem}.parquet"),
                }
            )
    (feature_sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    artifacts_path = output_dir / "post_nmf_artifacts.json"
    artifacts = {}
    if artifacts_path.exists():
        artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    artifacts.update(
        {
            "targets": str(output_dir / "targets_y.parquet"),
            "groups": str(output_dir / "groups.parquet"),
            "combined_features": str(output_dir / "combined_features_filtered.parquet"),
            "reduced_features": str(output_dir / "reduced_features_final_15.parquet"),
        }
    )
    artifacts_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild classifier inputs at FOV level from post-NMF outputs.")
    parser.add_argument("--output-dir", required=True, help="Run outputs directory.")
    parser.add_argument("--default-niche-gene-count", type=int, default=20)
    args = parser.parse_args()
    build_inputs(Path(args.output_dir), default_group_count=args.default_niche_gene_count)


if __name__ == "__main__":
    main()
