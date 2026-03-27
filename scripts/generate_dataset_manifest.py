#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def detect_gene_id_format(var_names) -> str:
    if not var_names:
        return "unknown"
    sample = list(var_names[:100])
    if any(name.startswith("ENSG") for name in sample):
        return "ensembl"
    if all(re.match(r"^[A-Z0-9_.-]+$", name or "") for name in sample):
        return "symbol"
    return "unknown"


def load_h5ad_schema(path: Path) -> dict:
    try:
        import anndata as ad
    except ImportError as exc:
        raise SystemExit("anndata is required to read h5ad files. Activate the pipeline conda env.") from exc
    adata = ad.read_h5ad(path)
    obs_keys = list(adata.obs_keys())
    has_raw_counts = adata.raw is not None
    gene_id_format = detect_gene_id_format(adata.var_names)
    coordinate_columns = ["CenterX_global_px", "CenterY_global_px"]
    morphology_columns = ["Area", "Width", "Height"]
    nmf_columns = ["NMF_factor", "dominant_nmf_factor"]
    return {
        "obs_keys": obs_keys,
        "has_raw_counts": has_raw_counts,
        "gene_id_format": gene_id_format,
        "has_spatial_coordinates": all(key in obs_keys for key in coordinate_columns),
        "has_morphology": ("Area" in obs_keys) or ("Width" in obs_keys and "Height" in obs_keys),
        "has_nmf_labels": any(key in obs_keys for key in nmf_columns),
    }


def load_metadata_columns(path: Path) -> list[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to read metadata files. Activate the pipeline conda env.") from exc
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def summarize_metadata_columns(columns: list[str]) -> dict:
    return {
        "has_spatial_coordinates": all(key in columns for key in ("CenterX_global_px", "CenterY_global_px")),
        "has_morphology": ("Area" in columns) or ("Width" in columns and "Height" in columns),
        "has_join_keys": ("unique_cell_id" in columns) or all(key in columns for key in ("fov", "cell_ID")),
    }


def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("datasets", [])
    if not isinstance(data, list):
        return []
    return data


def write_registry(path: Path, datasets: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(datasets, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a dataset registry entry with a cached schema manifest.")
    parser.add_argument("--id", required=True, help="Dataset ID (unique).")
    parser.add_argument("--label", required=True, help="Human-friendly dataset label.")
    parser.add_argument("--organ", required=True, help="Organ (e.g., colon).")
    parser.add_argument("--platform", required=True, help="Platform (e.g., cosmx).")
    parser.add_argument("--h5ad", required=True, help="Path to staged spatial h5ad.")
    parser.add_argument("--metadata", required=True, help="Path to spatial metadata CSV/TSV.")
    parser.add_argument("--recommended-preset", default="", help="Optional preset id.")
    parser.add_argument("--registry", default="", help="Path to datasets.json to update.")
    parser.add_argument("--output", default="", help="Write entry JSON to this path (optional).")
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad)
    metadata_path = Path(args.metadata)
    manifest = load_h5ad_schema(h5ad_path)
    metadata_columns = load_metadata_columns(metadata_path)

    entry = {
        "id": args.id,
        "label": args.label,
        "organ": args.organ,
        "platform": args.platform,
        "staged_path": h5ad_path.as_posix(),
        "cell_metadata_path": metadata_path.as_posix(),
        "recommended_preset": args.recommended_preset,
        "schema_manifest": manifest,
        "metadata_columns": metadata_columns,
        "metadata_manifest": summarize_metadata_columns(metadata_columns),
    }

    if args.output:
        Path(args.output).write_text(json.dumps(entry, indent=2), encoding="utf-8")
        print(f"Wrote entry to {args.output}")
    else:
        print(json.dumps(entry, indent=2))

    if args.registry:
        registry_path = Path(args.registry)
        datasets = load_registry(registry_path)
        datasets = [item for item in datasets if item.get("id") != args.id]
        datasets.append(entry)
        write_registry(registry_path, datasets)
        print(f"Updated registry {registry_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
