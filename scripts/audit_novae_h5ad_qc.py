#!/usr/bin/env python3
"""Read-only cross-tab QC for a baseline and annotated NOVAE H5AD pair.

This is intentionally a full-H5AD operation and must run in a CPU SLURM job
on shared storage, never on an HPG login node.  It does not filter, rewrite, or
otherwise mutate either input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

GRAPH_KEY = "spatial_connectivities"
VALID_KEY = "neighborhood_valid"
DEFAULT_DOMAIN_PREFIX = "novae_domains_res"


class H5ADQCCError(ValueError):
    """Input or output contract violation."""


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anndata is required for the H5AD QC audit") from exc
    return ad


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        os.close(fd)
        frame.to_csv(name, index=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _obs_ids(adata: Any, label: str) -> list[str]:
    ids = [str(value) for value in adata.obs_names]
    if len(ids) != len(set(ids)):
        raise H5ADQCCError(f"{label} obs IDs are not unique")
    return ids


def _validate_bool_mask(values: Iterable[Any], key: str) -> np.ndarray:
    result: list[bool] = []
    for value in values:
        if _missing(value) or not isinstance(value, (bool, np.bool_)):
            raise H5ADQCCError(f"obs[{key!r}] must contain only non-missing booleans")
        result.append(bool(value))
    return np.asarray(result, dtype=bool)


def _assigned(value: Any) -> bool:
    if _missing(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def validate_raw_counts(matrix: Any) -> None:
    """Validate the pilot's raw-count contract without densifying sparse X."""
    if getattr(matrix, "ndim", None) != 2:
        raise H5ADQCCError("source X must be a two-dimensional raw-count matrix")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    try:
        finite = bool(np.isfinite(values).all())
        nonnegative = bool((values >= 0).all())
        integer_like = bool(np.all(np.isclose(values, np.rint(values), rtol=0, atol=1e-8)))
    except (TypeError, ValueError) as exc:
        raise H5ADQCCError("source X must contain numeric raw counts") from exc
    if not finite:
        raise H5ADQCCError("source X raw counts contain non-finite values")
    if not nonnegative:
        raise H5ADQCCError("source X raw counts contain negative values")
    if not integer_like:
        raise H5ADQCCError("source X raw counts contain fractional values")


def _row_sums(matrix: Any) -> np.ndarray:
    validate_raw_counts(matrix)
    try:
        values = np.asarray(matrix.sum(axis=1)).ravel() if sparse.issparse(matrix) else np.asarray(matrix).sum(axis=1)
    except (TypeError, ValueError) as exc:
        raise H5ADQCCError("source X must be a numeric two-dimensional matrix") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise H5ADQCCError("source X row sums are non-finite")
    return values.astype(float, copy=False)


def validate_domain_consistency(adata: Any, domain_columns: Iterable[str], valid: np.ndarray, obs_ids: list[str]) -> None:
    """Require NOVAE's valid/invalid NA contract for every resolution."""
    for domain in domain_columns:
        assigned = np.asarray([_assigned(value) for value in adata.obs[domain].tolist()], dtype=bool)
        missing_valid = np.flatnonzero(valid & ~assigned)
        assigned_invalid = np.flatnonzero(~valid & assigned)
        if len(missing_valid) or len(assigned_invalid):
            valid_ids = [obs_ids[index] for index in missing_valid[:10]]
            invalid_ids = [obs_ids[index] for index in assigned_invalid[:10]]
            raise H5ADQCCError(
                f"domain {domain!r} violates neighborhood_valid contract: "
                f"valid_without_assignment={len(missing_valid)} sample IDs={valid_ids}; "
                f"invalid_with_assignment={len(assigned_invalid)} sample IDs={invalid_ids}"
            )


def _domain_columns(obs: Any, requested: str | None, prefix: str) -> list[str]:
    if requested:
        columns = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        columns = sorted(column for column in obs.columns if column.startswith(prefix))
        if "novae_leaves" in obs.columns and not columns:
            columns = ["novae_leaves"]
    if not columns:
        raise H5ADQCCError(f"no domain columns found with prefix {prefix!r}")
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise H5ADQCCError(f"required domain columns are missing: {missing}")
    return columns


def _graph_metrics(adata: Any, slide_values: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if GRAPH_KEY not in adata.obsp:
        raise H5ADQCCError(f"obsp[{GRAPH_KEY!r}] is missing")
    graph = adata.obsp[GRAPH_KEY]
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)
    graph = graph.tocsr().copy()
    if graph.shape != (adata.n_obs, adata.n_obs):
        raise H5ADQCCError("spatial graph shape does not match observations")
    try:
        finite_graph = bool(np.isfinite(graph.data).all())
    except TypeError as exc:
        raise H5ADQCCError("spatial graph must contain numeric values") from exc
    if not finite_graph:
        raise H5ADQCCError("spatial graph contains non-finite values")
    graph.eliminate_zeros()
    rows, cols = graph.nonzero()
    cross = np.asarray(slide_values)[rows] != np.asarray(slide_values)[cols]
    if np.any(cross):
        raise H5ADQCCError(f"spatial graph has {int(np.count_nonzero(cross))} cross-slide edges")
    undirected = graph.maximum(graph.T).tocsr()
    undirected.eliminate_zeros()
    degree = np.asarray(undirected.getnnz(axis=1)).ravel().astype(int)
    component_size = np.zeros(adata.n_obs, dtype=int)
    summaries: list[dict[str, Any]] = []
    slides = sorted(set(slide_values))
    for slide in slides:
        indices = np.flatnonzero(np.asarray(slide_values) == slide)
        sub = undirected[indices][:, indices]
        if len(indices):
            count, labels = connected_components(sub, directed=False, return_labels=True)
            sizes = np.bincount(labels, minlength=count)
            component_size[indices] = sizes[labels]
        else:
            count = 0
        summaries.append({
            "slide": slide, "spots": int(len(indices)),
            "graph_edges_undirected": int(sparse.triu(sub, k=1).nnz),
            "connected_components": int(count),
            "zero_degree": int(np.count_nonzero(degree[indices] == 0)),
        })
    return degree, component_size, summaries


def _coordinates(adata: Any) -> tuple[np.ndarray, list[str]]:
    if "spatial" not in adata.obsm:
        return np.full((adata.n_obs, 2), np.nan), ["spatial_x", "spatial_y"]
    values = np.asarray(adata.obsm["spatial"])
    if values.ndim != 2 or values.shape[0] != adata.n_obs or values.shape[1] < 2:
        raise H5ADQCCError("obsm['spatial'] must have at least two row-aligned columns")
    values = values[:, :2].astype(float, copy=False)
    if not np.isfinite(values).all():
        raise H5ADQCCError("obsm['spatial'] contains non-finite coordinates")
    return values, ["spatial_x", "spatial_y"]


def run_audit(source_h5ad: str | Path, annotated_h5ad: str | Path, output_dir: str | Path,
              *, slide_key: str = "sample_id", domain_columns: str | None = None,
              domain_prefix: str = DEFAULT_DOMAIN_PREFIX) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise H5ADQCCError(f"refusing existing output directory: {output}")
    source_path, annotated_path = Path(source_h5ad), Path(annotated_h5ad)
    if not source_path.is_file() or not annotated_path.is_file():
        raise H5ADQCCError("both H5AD input paths must be existing files")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=parent))
    try:
        ad = _require_anndata()
        source = ad.read_h5ad(source_path, backed=None)
        annotated = ad.read_h5ad(annotated_path, backed=None)
        source_ids = _obs_ids(source, "source")
        annotated_ids = _obs_ids(annotated, "annotated")
        if source_ids != annotated_ids:
            raise H5ADQCCError("source and annotated obs IDs are not exactly aligned in order")
        if slide_key not in annotated.obs:
            raise H5ADQCCError(f"annotated obs key {slide_key!r} is missing")
        raw_slides = annotated.obs[slide_key].tolist()
        missing_slide_indices = [
            index for index, value in enumerate(raw_slides)
            if _missing(value) or not str(value).strip() or str(value).strip().lower() == "nan"
        ]
        if missing_slide_indices:
            ids = [source_ids[index] for index in missing_slide_indices[:10]]
            raise H5ADQCCError(f"annotated obs key {slide_key!r} contains missing values; sample IDs={ids}")
        slides = [str(value).strip() for value in raw_slides]
        if VALID_KEY not in annotated.obs:
            raise H5ADQCCError(f"annotated obs key {VALID_KEY!r} is missing")
        valid = _validate_bool_mask(annotated.obs[VALID_KEY].tolist(), VALID_KEY)
        domains = _domain_columns(annotated.obs, domain_columns, domain_prefix)
        validate_domain_consistency(annotated, domains, valid, source_ids)
        degree, component_size, slide_summaries = _graph_metrics(annotated, slides)
        row_sums = _row_sums(source.X)
        coords, coordinate_names = _coordinates(annotated)
        common = {
            "obs_id": source_ids,
            "slide": slides,
            "row_sum": row_sums,
            "degree": degree,
            "component_size": component_size,
            "neighborhood_valid": valid,
            "in_tissue": source.obs["in_tissue"].tolist() if "in_tissue" in source.obs else [pd.NA] * source.n_obs,
            coordinate_names[0]: coords[:, 0], coordinate_names[1]: coords[:, 1],
        }
        for domain in domains:
            common[domain] = annotated.obs[domain].tolist()
        frame = pd.DataFrame(common)
        frame["zero_count"] = frame["row_sum"] == 0
        frame["zero_degree"] = frame["degree"] == 0
        zero_spots = frame.loc[frame["zero_count"]].copy()
        invalid = frame.loc[~frame["neighborhood_valid"]].copy()
        combos = [(zero, degree_zero, neighborhood_valid) for zero in (False, True) for degree_zero in (False, True) for neighborhood_valid in (False, True)]
        cross_rows = []
        for zero, degree_zero, neighborhood_valid in combos:
            selected = frame["zero_count"].eq(zero) & frame["zero_degree"].eq(degree_zero) & frame["neighborhood_valid"].eq(neighborhood_valid)
            cross_rows.append({"zero_count": zero, "zero_degree": degree_zero, "neighborhood_valid": neighborhood_valid, "count": int(selected.sum())})
        cross_tab = pd.DataFrame(cross_rows)
        slide_frame = pd.DataFrame(slide_summaries).set_index("slide")
        grouped = frame.groupby("slide", sort=True, dropna=False)
        slide_frame["zero_count"] = grouped["zero_count"].sum().astype(int)
        slide_frame["valid_neighborhood"] = grouped["neighborhood_valid"].sum().astype(int)
        slide_frame["invalid_neighborhood"] = (grouped.size() - slide_frame["valid_neighborhood"]).astype(int)
        for domain in domains:
            assigned = frame.assign(_assigned=frame[domain].map(_assigned)).groupby("slide", sort=True)["_assigned"].sum()
            slide_frame[f"assigned_{domain}"] = assigned.astype(int)
        slide_frame = slide_frame.reset_index()
        _atomic_csv(zero_spots, staging / "zero_count_spots.csv")
        _atomic_csv(invalid, staging / "invalid_neighborhoods.csv")
        _atomic_csv(cross_tab, staging / "zero_count_x_zero_degree_x_validity.csv")
        _atomic_csv(slide_frame, staging / "per_slide_summary.csv")
        payload = {
            "scope": "read-only full H5AD NOVAE QC cross-tab",
            "source_h5ad": str(source_path.resolve()), "annotated_h5ad": str(annotated_path.resolve()),
            "source_obs": int(source.n_obs), "annotated_obs": int(annotated.n_obs),
            "slide_key": slide_key, "domain_columns": domains,
            "source_x_contract": "finite_nonnegative_integer_like_raw_counts",
            "zero_count": int(frame["zero_count"].sum()),
            "zero_count_obs_ids": [str(value) for value in zero_spots["obs_id"].tolist()],
            "zero_degree": int(frame["zero_degree"].sum()),
            "invalid_neighborhood": int((~frame["neighborhood_valid"]).sum()),
            "zero_count_x_zero_degree_x_validity": cross_rows,
            "graph_key": GRAPH_KEY, "neighborhood_valid_key": VALID_KEY,
            "outputs": ["zero_count_spots.csv", "invalid_neighborhoods.csv", "zero_count_x_zero_degree_x_validity.csv", "per_slide_summary.csv"],
            "per_slide": slide_frame.to_dict(orient="records"),
            "caveat": "No rows were filtered and no H5AD input was mutated. Missing domain values are preserved as NA. No calibration correction or sensitivity inference is selected or applied.",
        }
        _atomic_json(payload, staging / "novae_h5ad_qc.json")
        staging.replace(output)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", required=True, type=Path)
    parser.add_argument("--annotated-h5ad", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--slide-key", default="sample_id")
    parser.add_argument("--domain-columns", help="comma-separated obs domain columns; default discovers novae_domains_res*")
    parser.add_argument("--domain-prefix", default=DEFAULT_DOMAIN_PREFIX)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run_audit(args.source_h5ad, args.annotated_h5ad, args.output_dir, slide_key=args.slide_key, domain_columns=args.domain_columns, domain_prefix=args.domain_prefix)
    except (OSError, H5ADQCCError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
