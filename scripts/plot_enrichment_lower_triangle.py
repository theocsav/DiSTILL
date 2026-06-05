#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot lower-triangle niche-to-niche enrichment score matrices by class "
            "from a per-FOV enrichment matrix."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Path to FOV enrichment CSV with disease labels.")
    parser.add_argument("--output-dir", required=True, help="Directory for output figures and tables.")
    parser.add_argument(
        "--annot-decimals",
        type=int,
        default=2,
        help="Number of decimals to show inside heatmap cells.",
    )
    return parser.parse_args()


def build_mean_matrix(df: pd.DataFrame, niche_ids: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=niche_ids, columns=niche_ids, dtype=float)
    for i in niche_ids:
        for j in niche_ids:
            col_ij = f"enrichment_{i}-{j}"
            col_ji = f"enrichment_{j}-{i}"
            if i == j and col_ij in df.columns:
                matrix.loc[i, j] = float(df[col_ij].mean())
            else:
                vals = []
                if col_ij in df.columns:
                    vals.append(df[col_ij].mean())
                if col_ji in df.columns:
                    vals.append(df[col_ji].mean())
                if vals:
                    matrix.loc[i, j] = float(np.mean(vals))
    return matrix


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    disease_col = "Disease/Health State" if "Disease/Health State" in df.columns else "Disease_State"
    if disease_col not in df.columns:
        raise KeyError("Could not find disease label column.")

    enrichment_cols = [c for c in df.columns if c.startswith("enrichment_")]
    niche_ids = sorted(
        {
            part
            for col in enrichment_cols
            for part in col.replace("enrichment_", "").split("-")
        },
        key=lambda x: (len(str(x)), str(x)),
    )

    class_labels = list(df[disease_col].dropna().astype(str).drop_duplicates())
    matrices: dict[str, pd.DataFrame] = {}
    for label in class_labels:
        sub = df[df[disease_col].astype(str) == label].copy()
        matrices[label] = build_mean_matrix(sub, niche_ids)
        matrices[label].to_csv(output_dir / f"{label}_mean_enrichment_matrix.csv")

    vmax = max(
        abs(val)
        for mat in matrices.values()
        for val in np.ravel(mat.to_numpy(dtype=float))
        if np.isfinite(val)
    )
    vmax = max(vmax, 1e-6)

    n = len(class_labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    mask = np.triu(np.ones((len(niche_ids), len(niche_ids)), dtype=bool), k=1)

    for ax, label in zip(axes[0], class_labels):
        mat = matrices[label]
        sns.heatmap(
            mat,
            mask=mask,
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=f".{args.annot_decimals}f",
            annot_kws={"size": 9},
            cbar=True,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("Niche")
        ax.set_ylabel("Niche")

    fig.suptitle("Lower-triangle niche enrichment score matrices by class", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "lower_triangle_enrichment_by_class.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
