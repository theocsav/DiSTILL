#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "cross_organ_comparison"

KIDNEY_TABLES = ROOT / "kidney_dataset" / "new_report" / "tables"
KIDNEY_MLP = ROOT / "kidney_dataset" / "new_report" / "MLP_44Features" / "mlp_results.txt"

SKIN_TABLES = ROOT / "skin_visium_manuscript_package" / "tables"
SKIN_MLP = SKIN_TABLES / "mlp_results.txt"


def norm_niche(value: object) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def niche_sort_key(value: object) -> tuple[int, object]:
    text = norm_niche(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def top_celltypes(df: pd.DataFrame, niche_col: str, topn: int = 5) -> pd.DataFrame:
    tmp = df.copy()
    tmp[niche_col] = tmp[niche_col].map(norm_niche)
    records: list[dict[str, object]] = []
    for _, row in tmp.iterrows():
        niche = str(row[niche_col])
        values = row.drop(labels=[niche_col])
        values = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
        top = values.head(topn)
        record: dict[str, object] = {"niche": niche}
        for i, (name, val) in enumerate(zip(top.index, top.values), start=1):
            record[f"top_celltype_{i}"] = str(name)
            record[f"top_celltype_{i}_prop"] = float(val)
        records.append(record)
    return pd.DataFrame(records)


def top_genes_from_significant_features(df: pd.DataFrame, topn: int = 10) -> pd.DataFrame:
    tmp = df.copy()
    tmp["niche"] = tmp["niche"].map(norm_niche)
    tmp = tmp.sort_values(["niche", "p_adj", "mi_score"], ascending=[True, True, False])
    return tmp.groupby("niche", sort=False).head(topn).copy()


def top_genes_from_rank_table(df: pd.DataFrame, topn: int = 10) -> pd.DataFrame:
    tmp = df.copy()
    tmp["niche"] = tmp["niche"].map(norm_niche)
    tmp = tmp.sort_values(["niche", "rank"], ascending=[True, True])
    return tmp.groupby("niche", sort=False).head(topn).copy()


def build_kidney_class_summary(comp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_df = comp.melt(
        id_vars=["field_of_view", "Disease/Health State"],
        var_name="niche",
        value_name="niche_proportion",
    )
    long_df["niche"] = long_df["niche"].map(norm_niche)
    summary = (
        long_df.groupby(["Disease/Health State", "niche"])["niche_proportion"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))

    classes = list(long_df["Disease/Health State"].drop_duplicates())
    rows: list[dict[str, object]] = []
    if len(classes) >= 2:
        class_a, class_b = classes[0], classes[1]
        for niche, sub in long_df.groupby("niche", sort=False):
            x = sub.loc[sub["Disease/Health State"] == class_a, "niche_proportion"].to_numpy()
            y = sub.loc[sub["Disease/Health State"] == class_b, "niche_proportion"].to_numpy()
            rows.append(
                {
                    "niche": niche,
                    "class_1": class_a,
                    "class_2": class_b,
                    "mean_class_1": float(np.mean(x)),
                    "mean_class_2": float(np.mean(y)),
                    "delta_class2_minus_class1": float(np.mean(y) - np.mean(x)),
                    "dominant_in": class_b if np.mean(y) > np.mean(x) else class_a,
                }
            )
    comparisons = pd.DataFrame(rows).sort_values("delta_class2_minus_class1")
    return summary, comparisons


def top_enrichment(df: pd.DataFrame, topn: int = 15) -> pd.DataFrame:
    tmp = df.copy()
    tmp["pair"] = tmp["feature"].str.replace("enrichment_", "", regex=False)
    return tmp.sort_values(["p_adj", "mi_score"], ascending=[True, False]).head(topn).copy()


def safe_get_top_genes(df: pd.DataFrame, niche: str, gene_col: str = "gene", topn: int = 5) -> str:
    sub = df[df["niche"].astype(str) == str(niche)]
    if gene_col not in sub.columns:
        return ""
    return ", ".join(sub[gene_col].astype(str).head(topn).tolist())


def build_mapping_table(
    kidney_ct_top: pd.DataFrame,
    skin_ct_top: pd.DataFrame,
    kidney_top_genes: pd.DataFrame,
    skin_top_genes: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        {
            "skin_niche": "6",
            "kidney_niche": "5",
            "match_type": "shared fibrotic/stromal remodeling niche",
            "confidence": "high",
            "shared_program": "ECM remodeling / fibroblast activation",
            "rationale": "Both niches are stromal-fibroblast dominant and enriched in collagen/remodeling genes consistent with fibrosis.",
            "shared_genes_or_markers": "COL1A1, COL3A1",
        },
        {
            "skin_niche": "4",
            "kidney_niche": "9",
            "match_type": "injury-associated remodeling niche",
            "confidence": "medium",
            "shared_program": "activated remodeling / tissue injury response",
            "rationale": "Skin niche 4 is fibroblast-heavy and disease-enriched; kidney niche 9 is strongly disease-enriched with injury/remodeling markers, suggesting related but non-identical SSc remodeling states.",
            "shared_genes_or_markers": "COL1A1, TIMP1 / matrix-remodeling context",
        },
        {
            "skin_niche": "1",
            "kidney_niche": "7",
            "match_type": "mesenchymal/perivascular remodeling niche",
            "confidence": "low-medium",
            "shared_program": "mesenchymal support / vascular-adjacent remodeling",
            "rationale": "Skin niche 1 is fibroblast-dominant; kidney niche 7 is VSMC/perivascular dominant. They likely reflect organ-specific mesenchymal remodeling compartments rather than exact one-to-one analogs.",
            "shared_genes_or_markers": "stromal/perivascular program, limited direct gene overlap",
        },
        {
            "skin_niche": "0",
            "kidney_niche": "8",
            "match_type": "healthy structural epithelial niche",
            "confidence": "low",
            "shared_program": "organ-specific healthy parenchymal architecture",
            "rationale": "Skin niche 0 is keratinocyte/epidermal; kidney niche 8 is healthy tubular/parenchymal. They are not biologically identical, but each represents dominant healthy tissue structure.",
            "shared_genes_or_markers": "no direct gene overlap expected",
        },
        {
            "skin_niche": "3",
            "kidney_niche": "4",
            "match_type": "healthy epithelial differentiation niche",
            "confidence": "low",
            "shared_program": "non-disease structural epithelial state",
            "rationale": "Skin niche 3 is keratinocyte-differentiation heavy; kidney niche 4 appears healthier epithelial/tubular. This is a conceptual rather than molecular match.",
            "shared_genes_or_markers": "epithelial identity, organ-specific markers",
        },
    ]

    for record in records:
        skin_niche = record["skin_niche"]
        kidney_niche = record["kidney_niche"]

        skin_row = skin_ct_top[skin_ct_top["niche"] == skin_niche].iloc[0]
        kidney_row = kidney_ct_top[kidney_ct_top["niche"] == kidney_niche].iloc[0]

        record["skin_top_celltypes"] = ", ".join(
            str(skin_row.get(f"top_celltype_{i}", "")) for i in range(1, 4) if pd.notna(skin_row.get(f"top_celltype_{i}", np.nan))
        )
        record["kidney_top_celltypes"] = ", ".join(
            str(kidney_row.get(f"top_celltype_{i}", "")) for i in range(1, 4) if pd.notna(kidney_row.get(f"top_celltype_{i}", np.nan))
        )
        record["skin_top_genes"] = safe_get_top_genes(skin_top_genes, skin_niche)
        record["kidney_top_genes"] = safe_get_top_genes(kidney_top_genes, kidney_niche)

    return pd.DataFrame(records)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)

    kidney_comp = pd.read_csv(KIDNEY_TABLES / "rcausal_mgm__NicheCompositions__niche_compositions_percent.csv")
    kidney_ct = pd.read_csv(KIDNEY_TABLES / "NMF_Niche_CellType_Proportions_Normalized.csv")
    kidney_genes = pd.read_csv(KIDNEY_TABLES / "niche_gene_significant_features.csv")
    kidney_enrich = pd.read_csv(KIDNEY_TABLES / "enrichment_significant_features.csv")

    skin_ct = pd.read_csv(SKIN_TABLES / "NMF_Niche_CellType_Proportions_Normalized.csv")
    skin_comp_summary = pd.read_csv(SKIN_TABLES / "skin_niche_proportions_per_class_summary.csv")
    skin_comp_cmp = pd.read_csv(SKIN_TABLES / "skin_niche_proportion_class_comparisons.csv")
    skin_genes = pd.read_csv(SKIN_TABLES / "top_50_genes_per_niche.csv")
    skin_enrich = pd.read_csv(SKIN_TABLES / "enrichment_significant_features.csv")

    kidney_summary, kidney_cmp = build_kidney_class_summary(kidney_comp)
    kidney_summary.to_csv(OUTDIR / "kidney_niche_proportions_per_class_summary.csv", index=False)
    kidney_cmp.to_csv(OUTDIR / "kidney_niche_proportion_class_comparisons_simple.csv", index=False)

    kidney_ct_top = top_celltypes(kidney_ct, niche_col="dominant_nmf_factor")
    skin_ct_top = top_celltypes(skin_ct, niche_col="dominant_nmf_factor")
    kidney_ct_top.to_csv(OUTDIR / "kidney_top_celltypes_per_niche.csv", index=False)
    skin_ct_top.to_csv(OUTDIR / "skin_top_celltypes_per_niche.csv", index=False)

    kidney_top_genes = top_genes_from_significant_features(kidney_genes)
    skin_top_genes = top_genes_from_rank_table(skin_genes)
    kidney_top_genes.to_csv(OUTDIR / "kidney_top10_genes_per_niche_from_significant_features.csv", index=False)
    skin_top_genes.to_csv(OUTDIR / "skin_top10_genes_per_niche.csv", index=False)

    kidney_enrich_top = top_enrichment(kidney_enrich)
    skin_enrich_top = top_enrichment(skin_enrich)
    kidney_enrich_top.to_csv(OUTDIR / "kidney_top_enrichment_features.csv", index=False)
    skin_enrich_top.to_csv(OUTDIR / "skin_top_enrichment_features.csv", index=False)

    mapping = build_mapping_table(kidney_ct_top, skin_ct_top, kidney_top_genes, skin_top_genes)
    mapping.to_csv(OUTDIR / "kidney_skin_niche_mapping_table.csv", index=False)

    kidney_ssc = kidney_cmp.sort_values("delta_class2_minus_class1", ascending=False).head(3)
    kidney_healthy = kidney_cmp.sort_values("delta_class2_minus_class1", ascending=True).head(3)
    skin_cmp2 = skin_comp_cmp.copy()
    skin_cmp2["niche"] = skin_cmp2["niche"].map(norm_niche)
    skin_cmp2["delta_class2_minus_class1"] = skin_cmp2["mean_class_2"] - skin_cmp2["mean_class_1"]
    skin_ssc = skin_cmp2.sort_values("delta_class2_minus_class1", ascending=False).head(3)
    skin_healthy = skin_cmp2.sort_values("delta_class2_minus_class1", ascending=True).head(3)

    lines: list[str] = []
    lines.append("# Kidney vs Skin SSc niche comparison")
    lines.append("")
    lines.append("## Canonical provenance")
    lines.append(f"- Kidney source set: `{KIDNEY_TABLES.relative_to(ROOT.parent)}`")
    lines.append(f"- Skin source set: `{SKIN_TABLES.relative_to(ROOT.parent)}`")
    lines.append("")
    lines.append("## Dataset-level summary")
    lines.append(f"- Kidney niches: {sorted(kidney_ct_top['niche'].tolist(), key=niche_sort_key)}")
    lines.append(f"- Skin niches: {sorted(skin_ct_top['niche'].tolist(), key=niche_sort_key)}")
    lines.append("")
    lines.append("## Disease-enriched niches")
    lines.append("### Kidney: niches increased in systemic_sclerosis vs healthy")
    for _, row in kidney_ssc.iterrows():
        niche = str(row["niche"])
        ct = kidney_ct_top[kidney_ct_top["niche"] == niche].iloc[0]
        genes = safe_get_top_genes(kidney_top_genes, niche)
        lines.append(
            f"- Kidney niche {niche}: Δ={row['delta_class2_minus_class1']:.3f}; "
            f"top cell types: {ct.get('top_celltype_1', '?')}, {ct.get('top_celltype_2', '?')}, {ct.get('top_celltype_3', '?')}; "
            f"top genes: {genes}"
        )
    lines.append("### Skin: niches increased in systemic_sclerosis vs healthy")
    for _, row in skin_ssc.iterrows():
        niche = str(row["niche"])
        ct = skin_ct_top[skin_ct_top["niche"] == niche].iloc[0]
        genes = safe_get_top_genes(skin_top_genes, niche)
        lines.append(
            f"- Skin niche {niche}: Δ={row['delta_class2_minus_class1']:.3f}; "
            f"top cell types: {ct.get('top_celltype_1', '?')}, {ct.get('top_celltype_2', '?')}, {ct.get('top_celltype_3', '?')}; "
            f"top genes: {genes}"
        )
    lines.append("")
    lines.append("## Healthy-enriched niches")
    lines.append("### Kidney")
    for _, row in kidney_healthy.iterrows():
        niche = str(row["niche"])
        ct = kidney_ct_top[kidney_ct_top["niche"] == niche].iloc[0]
        genes = safe_get_top_genes(kidney_top_genes, niche)
        lines.append(
            f"- Kidney niche {niche}: Δ={row['delta_class2_minus_class1']:.3f}; "
            f"top cell types: {ct.get('top_celltype_1', '?')}, {ct.get('top_celltype_2', '?')}, {ct.get('top_celltype_3', '?')}; "
            f"top genes: {genes}"
        )
    lines.append("### Skin")
    for _, row in skin_healthy.iterrows():
        niche = str(row["niche"])
        ct = skin_ct_top[skin_ct_top["niche"] == niche].iloc[0]
        genes = safe_get_top_genes(skin_top_genes, niche)
        lines.append(
            f"- Skin niche {niche}: Δ={row['delta_class2_minus_class1']:.3f}; "
            f"top cell types: {ct.get('top_celltype_1', '?')}, {ct.get('top_celltype_2', '?')}, {ct.get('top_celltype_3', '?')}; "
            f"top genes: {genes}"
        )
    lines.append("")
    lines.append("## Top enrichment differences")
    lines.append(f"- Kidney top enrichment features: {', '.join(kidney_enrich_top['feature'].head(6).tolist())}")
    lines.append(f"- Skin top enrichment features: {', '.join(skin_enrich_top['feature'].head(6).tolist())}")
    lines.append("")
    lines.append("## Cross-organ interpretation (canonical-source pass)")
    lines.append("- Both organs show disease-associated stromal / mesenchymal remodeling, but through different organ-specific niche architectures.")
    lines.append("- Kidney systemic_sclerosis-enriched niches are dominated by injury / stromal / fibrovascular programs, with recurrent genes such as MMP7, IFI6, TIMP1, and COL1A1 in significant niche-gene features.")
    lines.append("- Skin systemic_sclerosis-enriched niches are dominated by fibroblast-rich programs, especially niches 6 and 4, with matrix-remodeling fibroblast states more prominent than immune-only niches.")
    lines.append("- Healthy skin niches are strongly keratinocyte / epidermal, whereas healthy kidney niches are tubular / endothelial, so the baseline tissue-specific identities differ even when the disease trend points toward remodeling in both organs.")
    lines.append("- The strongest shared conclusion is not that the same numbered niche recurs across organs, but that systemic sclerosis pushes both organs toward spatially organized stromal remodeling / fibrosis-like states.")
    lines.append("- Neighborhood enrichment changes are present in both organs, but the top niche-pair shifts are not identical, which argues for shared disease direction with organ-specific spatial implementation.")
    lines.append("")
    lines.append("## Candidate conclusion")
    lines.append("- A publishable conclusion could be that systemic sclerosis has a cross-organ spatial signature centered on stromal/fibrotic remodeling, but the exact niche composition and niche-neighborhood structure remain strongly organ dependent.")
    (OUTDIR / "kidney_skin_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "kidney_source": str(KIDNEY_TABLES.relative_to(ROOT)),
        "skin_source": str(SKIN_TABLES.relative_to(ROOT)),
        "kidney_mlp": str(KIDNEY_MLP.relative_to(ROOT)),
        "skin_mlp": str(SKIN_MLP.relative_to(ROOT)),
        "outputs": [
            "kidney_niche_proportions_per_class_summary.csv",
            "kidney_niche_proportion_class_comparisons_simple.csv",
            "kidney_top_celltypes_per_niche.csv",
            "skin_top_celltypes_per_niche.csv",
            "kidney_top10_genes_per_niche_from_significant_features.csv",
            "skin_top10_genes_per_niche.csv",
            "kidney_top_enrichment_features.csv",
            "skin_top_enrichment_features.csv",
            "kidney_skin_niche_mapping_table.csv",
            "kidney_skin_comparison_summary.md",
        ],
    }
    (OUTDIR / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
