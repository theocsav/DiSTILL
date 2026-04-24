import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse


FACTOR_CANDIDATES = ("NMF_factor", "dominant_nmf_factor", "cell_type")
INVALID_FACTOR_LABELS = {"nan", "none", "unassigned"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export top genes per niche from an NMF-annotated h5ad. "
            "By default, uses adata.X and reports mean expression within each niche."
        )
    )
    parser.add_argument("--h5ad", required=True, help="Path to the NMF-annotated h5ad.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV outputs should be written.",
    )
    parser.add_argument(
        "--top-n",
        nargs="+",
        type=int,
        default=[20, 30, 50],
        help="Top-N cutoffs to export for each niche.",
    )
    parser.add_argument(
        "--factor-column",
        default=None,
        help="Obs column containing niche assignments. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--matrix",
        default="X",
        help=(
            "Expression source: X, raw, or layer:<layer_name>. "
            "Use a normalized/log-normalized matrix if normalized values are desired."
        ),
    )
    return parser.parse_args()


def choose_factor_column(obs: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in obs.columns:
            raise KeyError(f"Requested factor column not found: {requested}")
        return requested
    for candidate in FACTOR_CANDIDATES:
        if candidate in obs.columns:
            return candidate
    raise KeyError(
        "No niche-assignment column found. Tried: "
        + ", ".join(FACTOR_CANDIDATES)
    )


def choose_matrix(adata: ad.AnnData, matrix_spec: str):
    if matrix_spec == "X":
        return adata.X, pd.Index(adata.var_names.astype(str)), "X"
    if matrix_spec == "raw":
        if adata.raw is None:
            raise ValueError("Requested matrix=raw, but adata.raw is not present.")
        return adata.raw.X, pd.Index(adata.raw.var_names.astype(str)), "raw"
    if matrix_spec.startswith("layer:"):
        layer_name = matrix_spec.split(":", 1)[1]
        if layer_name not in adata.layers:
            raise KeyError(f"Requested layer not found: {layer_name}")
        return adata.layers[layer_name], pd.Index(adata.var_names.astype(str)), f"layer:{layer_name}"
    raise ValueError("matrix must be one of: X, raw, layer:<layer_name>")


def clean_factor_labels(values: pd.Series) -> pd.Series:
    labels = values.astype(str)
    invalid_mask = labels.str.lower().isin(INVALID_FACTOR_LABELS)
    return labels.where(~invalid_mask)


def sorted_factor_labels(labels: pd.Series) -> list[str]:
    unique_labels = [label for label in labels.dropna().unique()]

    def sort_key(label: str):
        return (0, int(label)) if label.isdigit() else (1, label)

    return sorted(unique_labels, key=sort_key)


def group_mean_expression(matrix, group_codes: np.ndarray, n_groups: int) -> np.ndarray:
    rows = np.arange(len(group_codes))
    selector = csr_matrix(
        (np.ones(len(group_codes), dtype=float), (group_codes, rows)),
        shape=(n_groups, len(group_codes)),
    )
    if issparse(matrix):
        grouped = selector @ matrix.tocsr()
        grouped = grouped.toarray()
    else:
        grouped = selector @ np.asarray(matrix)
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    return grouped / counts[:, None]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    factor_column = choose_factor_column(adata.obs, args.factor_column)
    matrix, gene_names, matrix_label = choose_matrix(adata, args.matrix)

    factor_labels = clean_factor_labels(adata.obs[factor_column])
    valid_mask = factor_labels.notna().to_numpy()
    if not valid_mask.any():
        raise ValueError("No valid niche assignments found after filtering invalid labels.")

    factor_labels = factor_labels.loc[valid_mask]
    matrix_valid = matrix[valid_mask]
    niche_order = sorted_factor_labels(factor_labels)
    category_map = {label: idx for idx, label in enumerate(niche_order)}
    group_codes = factor_labels.map(category_map).to_numpy(dtype=int)

    mean_expression = group_mean_expression(matrix_valid, group_codes, len(niche_order))
    niche_counts = pd.Series(factor_labels).value_counts().reindex(niche_order)

    top_ns = sorted(set(n for n in args.top_n if n > 0))
    if not top_ns:
        raise ValueError("At least one positive top-n cutoff is required.")

    all_records: list[dict] = []
    for niche_idx, niche_label in enumerate(niche_order):
        means = mean_expression[niche_idx]
        ranked_gene_idx = np.argsort(-means)
        for rank, gene_idx in enumerate(ranked_gene_idx, start=1):
            gene = str(gene_names[gene_idx])
            mean_value = float(means[gene_idx])
            record = {
                "niche": niche_label,
                "rank": rank,
                "gene": gene,
                "mean_expression": mean_value,
                "cells_in_niche": int(niche_counts.loc[niche_label]),
                "factor_column": factor_column,
                "matrix": matrix_label,
            }
            all_records.append(record)
            if rank >= max(top_ns):
                break

    all_df = pd.DataFrame(all_records)
    all_df.to_csv(output_dir / "top_niche_genes_long.csv", index=False)

    for top_n in top_ns:
        top_df = all_df[all_df["rank"] <= top_n].copy()
        top_df.to_csv(output_dir / f"top_{top_n}_genes_per_niche.csv", index=False)

    summary = pd.DataFrame(
        {
            "niche": niche_order,
            "cells_in_niche": [int(niche_counts.loc[niche]) for niche in niche_order],
        }
    )
    summary["matrix"] = matrix_label
    summary["factor_column"] = factor_column
    summary.to_csv(output_dir / "niche_gene_export_summary.csv", index=False)


if __name__ == "__main__":
    main()
