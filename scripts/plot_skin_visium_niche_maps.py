#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
]


def _pick_column(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"None of the expected columns were found: {candidates}")


def _sample_diversity(df: pd.DataFrame, patient_col: str, factor_col: str, disease_col: str) -> pd.DataFrame:
    rows = []
    for patient, sub in df.groupby(patient_col, sort=False):
        counts = sub[factor_col].astype(str).value_counts()
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        rows.append(
            {
                "patient": str(patient),
                "Disease_State": str(sub[disease_col].iloc[0]),
                "n_spots": int(len(sub)),
                "n_niches": int(counts.size),
                "entropy": entropy,
                "top_niche_fraction": float(probs.iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Disease_State", "entropy", "n_spots"], ascending=[True, False, False])


def _choose_samples(summary: pd.DataFrame, total_samples: int) -> list[str]:
    groups = []
    disease_groups = list(summary["Disease_State"].drop_duplicates())
    per_group = max(1, total_samples // max(1, len(disease_groups)))
    for disease in disease_groups:
        sub = summary[summary["Disease_State"] == disease].copy()
        groups.extend(sub.head(per_group)["patient"].tolist())
    if len(groups) < total_samples:
        remaining = summary[~summary["patient"].isin(groups)]
        groups.extend(remaining.head(total_samples - len(groups))["patient"].tolist())
    return groups[:total_samples]


def _build_color_map(labels: list[str]) -> dict[str, str]:
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Visium niche maps on tissue images from a completed skin run.")
    parser.add_argument("--post-nmf-obs", required=True, help="Path to post_nmf_obs.csv from the completed run.")
    parser.add_argument("--images-dir", required=True, help="Directory containing per-sample extracted Visium images.")
    parser.add_argument("--output-dir", required=True, help="Directory for output figures.")
    parser.add_argument("--sample-manifest", help="Optional sample manifest CSV with disease/sample metadata.")
    parser.add_argument("--samples", nargs="+", help="Optional explicit sample IDs to plot.")
    parser.add_argument("--n-samples", type=int, default=6, help="Number of representative samples to plot if --samples is omitted.")
    parser.add_argument("--point-size", type=float, default=14.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(args.post_nmf_obs)
    patient_col = _pick_column(list(obs.columns), ["patient", "Patient", "sample_id"])
    disease_col = _pick_column(list(obs.columns), ["disease_state", "Disease_State", "Disease/Health State"])
    factor_col = _pick_column(list(obs.columns), ["NMF_factor", "dominant_nmf_factor", "nmf_factor"])
    x_col = _pick_column(list(obs.columns), ["CenterX_global_px", "pxl_col_in_fullres", "CenterX_local_px"])
    y_col = _pick_column(list(obs.columns), ["CenterY_global_px", "pxl_row_in_fullres", "CenterY_local_px"])

    obs[patient_col] = obs[patient_col].astype(str)
    obs[disease_col] = obs[disease_col].astype(str)
    obs[factor_col] = obs[factor_col].astype(str)

    summary = _sample_diversity(obs, patient_col, factor_col, disease_col)
    summary.to_csv(output_dir / "sample_niche_diversity_summary.csv", index=False)

    if args.samples:
        selected_samples = [str(s) for s in args.samples]
    else:
        selected_samples = _choose_samples(summary, args.n_samples)

    selected_summary = summary[summary["patient"].isin(selected_samples)].copy()
    selected_summary.to_csv(output_dir / "selected_samples_summary.csv", index=False)

    labels = sorted(obs[factor_col].dropna().astype(str).unique(), key=lambda v: (len(v), v))
    color_map = _build_color_map(labels)

    ncols = 2
    nrows = ceil(len(selected_samples) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, sample in zip(axes_flat, selected_samples):
        sub = obs[obs[patient_col] == sample].copy()
        disease = str(sub[disease_col].iloc[0])
        image_path = Path(args.images_dir) / sample / "tissue_hires_image.png"
        if image_path.exists():
            image = plt.imread(image_path)
            ax.imshow(image)
        else:
            ax.set_facecolor("white")

        for niche in labels:
            niche_sub = sub[sub[factor_col] == niche]
            if niche_sub.empty:
                continue
            ax.scatter(
                niche_sub[x_col].astype(float),
                niche_sub[y_col].astype(float),
                s=args.point_size,
                c=color_map[niche],
                alpha=args.alpha,
                linewidths=0,
                label=f"N{niche}",
            )

        ax.set_title(f"{sample} ({disease})\nspots={len(sub)}, niches={sub[factor_col].nunique()}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(left=0)
        ax.set_ylim(ax.get_ylim()[::-1])

    for ax in axes_flat[len(selected_samples):]:
        ax.axis("off")

    legend_handles = [mpatches.Patch(color=color_map[label], label=f"N{label}") for label in labels]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(labels), 8), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "representative_skin_visium_niche_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for sample in selected_samples:
        sub = obs[obs[patient_col] == sample].copy()
        disease = str(sub[disease_col].iloc[0])
        image_path = Path(args.images_dir) / sample / "tissue_hires_image.png"
        plt.figure(figsize=(8, 8))
        if image_path.exists():
            image = plt.imread(image_path)
            plt.imshow(image)
        for niche in labels:
            niche_sub = sub[sub[factor_col] == niche]
            if niche_sub.empty:
                continue
            plt.scatter(
                niche_sub[x_col].astype(float),
                niche_sub[y_col].astype(float),
                s=args.point_size,
                c=color_map[niche],
                alpha=args.alpha,
                linewidths=0,
            )
        plt.title(f"{sample} ({disease})")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(left=0)
        plt.ylim(plt.ylim()[::-1])
        plt.tight_layout()
        plt.savefig(output_dir / f"{sample}_niche_map.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("Selected samples:", ", ".join(selected_samples))
    print(f"Wrote {output_dir / 'representative_skin_visium_niche_maps.png'}")


if __name__ == "__main__":
    main()
