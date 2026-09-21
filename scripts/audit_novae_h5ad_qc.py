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


def _slide_values(adata: Any, slide_key: str, label: str) -> list[str]:
    if slide_key not in adata.obs:
        raise H5ADQCCError(f"{label} obs key {slide_key!r} is missing")
    values: list[str] = []
    missing: list[int] = []
    for index, value in enumerate(adata.obs[slide_key].tolist()):
        if _missing(value) or not str(value).strip() or str(value).strip().lower() == "nan":
            missing.append(index)
        else:
            values.append(str(value).strip())
    if missing:
        ids = [str(adata.obs_names[index]) for index in missing[:10]]
        raise H5ADQCCError(f"{label} obs key {slide_key!r} contains missing values; sample IDs={ids}")
    return values


def _manifest_geometry(path: str | Path, observed_slides: list[str]) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise H5ADQCCError(f"sample manifest does not exist: {manifest_path}")
    try:
        frame = pd.read_json(manifest_path) if manifest_path.suffix.lower() == ".json" else pd.read_csv(manifest_path)
    except (OSError, ValueError) as exc:
        raise H5ADQCCError(f"could not read sample manifest: {manifest_path}") from exc
    required = {"sample_id", "spot_diameter_fullres"}
    missing = required - set(frame.columns)
    if missing:
        raise H5ADQCCError(f"sample manifest is missing columns: {sorted(missing)}")
    ids = []
    for value in frame["sample_id"].tolist():
        if _missing(value) or not str(value).strip() or str(value).strip().lower() == "nan":
            raise H5ADQCCError("sample manifest contains missing sample_id")
        ids.append(str(value).strip())
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise H5ADQCCError(f"sample manifest has duplicate sample_id entries: {duplicates}")
    observed = set(observed_slides)
    listed = set(ids)
    if listed != observed:
        raise H5ADQCCError(
            "sample manifest slides must exactly match observed slides; "
            f"missing={sorted(observed - listed)}, extra={sorted(listed - observed)}"
        )
    rows: dict[str, dict[str, Any]] = {}
    for index, sample_id in enumerate(ids):
        try:
            diameter = float(frame.iloc[index]["spot_diameter_fullres"])
        except (TypeError, ValueError) as exc:
            raise H5ADQCCError(f"invalid spot diameter for {sample_id!r}") from exc
        if not np.isfinite(diameter) or diameter <= 0:
            raise H5ADQCCError(f"spot diameter for {sample_id!r} must be finite and positive")
        row: dict[str, Any] = {"spot_diameter_fullres": diameter}
        for key in ("tissue_hires_scalef", "tissue_lowres_scalef"):
            if key in frame.columns:
                value = frame.iloc[index][key]
                if _missing(value) or str(value).strip() == "":
                    row[key] = None
                else:
                    try:
                        number = float(value)
                    except (TypeError, ValueError) as exc:
                        raise H5ADQCCError(f"manifest {key} for {sample_id!r} is not numeric") from exc
                    if not np.isfinite(number) or number <= 0:
                        raise H5ADQCCError(f"manifest {key} for {sample_id!r} must be finite and positive")
                    row[key] = number
            else:
                row[key] = None
        rows[sample_id] = row
    return rows


def _source_pixels(source: Any) -> tuple[np.ndarray, str]:
    if "spatial" in source.obsm:
        values = np.asarray(source.obsm["spatial"])
        source_name = "source.obsm['spatial']"
    else:
        pairs = (("CenterX", "CenterY"), ("CenterX_global_px", "CenterY_global_px"),
                 ("CenterX_local_px", "CenterY_local_px"))
        pair = next((candidate for candidate in pairs if set(candidate).issubset(source.obs.columns)), None)
        if pair is None:
            raise H5ADQCCError("source requires obsm['spatial'] or validated obs CenterX/CenterY pixel columns")
        values = source.obs[list(pair)].to_numpy()
        source_name = f"source.obs[{pair[0]!r},{pair[1]!r}]"
    if values.ndim != 2 or values.shape[0] != source.n_obs or values.shape[1] < 2:
        raise H5ADQCCError(f"{source_name} must have at least two row-aligned columns")
    try:
        pixels = values[:, :2].astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise H5ADQCCError(f"{source_name} must contain numeric pixel coordinates") from exc
    if not np.isfinite(pixels).all():
        raise H5ADQCCError(f"{source_name} contains non-finite coordinates")
    return pixels, source_name


def _source_geometry(source: Any, slides: list[str], manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"in_tissue", "array_row", "array_col"}
    missing = required - set(source.obs.columns)
    if missing:
        raise H5ADQCCError(f"source obs is missing geometry columns: {sorted(missing)}")
    pixels, pixel_source = _source_pixels(source)
    in_tissue: list[bool] = []
    lattice: list[tuple[int, int]] = []
    for index, value in enumerate(source.obs["in_tissue"].tolist()):
        if _missing(value):
            raise H5ADQCCError(f"source obs['in_tissue'] contains missing values at row {index + 1}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise H5ADQCCError("source obs['in_tissue'] must contain 0/1 values") from exc
        if not np.isfinite(number) or number not in (0, 1):
            raise H5ADQCCError("source obs['in_tissue'] must contain only finite 0/1 values")
        in_tissue.append(bool(number))
        try:
            row, col = float(source.obs.iloc[index]["array_row"]), float(source.obs.iloc[index]["array_col"])
        except (TypeError, ValueError) as exc:
            raise H5ADQCCError(f"source lattice values are not numeric at row {index + 1}") from exc
        if not np.isfinite(row) or not np.isfinite(col) or row != int(row) or col != int(col):
            raise H5ADQCCError(f"source lattice values must be finite integers at row {index + 1}")
        lattice.append((int(row), int(col)))
    results: list[dict[str, Any]] = []
    offsets = ((0, 2), (1, -1), (1, 1))
    for slide in sorted(set(slides)):
        indices = [index for index, value in enumerate(slides) if value == slide and in_tissue[index]]
        if not indices:
            raise H5ADQCCError(f"slide {slide!r} has no observed in-tissue spots")
        raw_barcodes = source.obs["barcode"].tolist() if "barcode" in source.obs else [str(value) for value in source.obs_names]
        ids = []
        for index in indices:
            barcode = raw_barcodes[index]
            if _missing(barcode) or not str(barcode).strip():
                raise H5ADQCCError(f"slide {slide!r} has a missing barcode")
            ids.append(str(barcode).strip())
        if len(ids) != len(set(ids)):
            raise H5ADQCCError(f"slide {slide!r} has duplicate barcodes")
        by_lattice = {lattice[index]: index for index in indices}
        if len(by_lattice) != len(indices):
            raise H5ADQCCError(f"slide {slide!r} has duplicate lattice positions")
        by_pixel = {(float(pixels[index, 0]), float(pixels[index, 1])): index for index in indices}
        if len(by_pixel) != len(indices):
            raise H5ADQCCError(f"slide {slide!r} has duplicate pixel coordinates")
        pairs: list[tuple[int, int]] = []
        for coord in sorted(by_lattice):
            for dr, dc in offsets:
                other = (coord[0] + dr, coord[1] + dc)
                if other in by_lattice:
                    pairs.append((by_lattice[coord], by_lattice[other]))
        if not pairs:
            raise H5ADQCCError(f"slide {slide!r} has no canonical lattice edges")
        distances = np.asarray([np.linalg.norm(pixels[left] - pixels[right]) for left, right in pairs], dtype=float)
        if not np.isfinite(distances).all() or np.any(distances <= 0):
            raise H5ADQCCError(f"slide {slide!r} has nonpositive or nonfinite pixel pitch")
        degree = {index: 0 for index in indices}
        adjacency = {index: set() for index in indices}
        for left, right in pairs:
            degree[left] += 1
            degree[right] += 1
            adjacency[left].add(right)
            adjacency[right].add(left)
        unseen = set(indices)
        components = 0
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
        q1, median, q3 = np.quantile(distances, [0.25, 0.5, 0.75], method="linear")
        diameter = manifest[slide]["spot_diameter_fullres"]
        current_scale = 55.0 / diameter
        nominal_scale = 100.0 / float(median)
        result = {
            "sample_id": slide, "observed_in_tissue_spots": len(indices),
            "canonical_lattice_edges": len(pairs), "canonical_lattice_edges_observed": len(pairs),
            "canonical_lattice_edges_in_tissue": len(pairs),
            "zero_degree": sum(value == 0 for value in degree.values()),
            "canonical_lattice_zero_degree_observed": sum(value == 0 for value in degree.values()),
            "lattice_zero_degree_in_tissue": sum(value == 0 for value in degree.values()),
            "connected_components": components, "canonical_lattice_components_observed": components,
            "lattice_connected_components_in_tissue": components,
            "pixel_pitch_min_px": float(distances.min()), "pixel_pitch_q1_px": float(q1),
            "pixel_pitch_median_px": float(median), "pixel_pitch_q3_px": float(q3),
            "pixel_pitch_max_px": float(distances.max()), "pixel_pitch_iqr_px": float(q3 - q1),
            "pixel_pitch_min": float(distances.min()), "pixel_pitch_q1": float(q1),
            "pixel_pitch_median": float(median), "pixel_pitch_q3": float(q3),
            "pixel_pitch_max": float(distances.max()), "pixel_pitch_iqr": float(q3 - q1),
            "spot_diameter_fullres_px": diameter,
            "current_55um_scale_um_per_pixel": current_scale,
            "current_55um_resulting_median_um": float(median * current_scale),
            "nominal_100um_scale_um_per_pixel": nominal_scale,
            "nominal_100um_array_pitch_scale_um_per_pixel": nominal_scale,
            "nominal_100um_implied_spot_diameter_um": float(diameter * nominal_scale),
            "implied_spot_diameter_under_nominal_pitch_um": float(diameter * nominal_scale),
            "nominal_to_current_scale_ratio": nominal_scale / current_scale,
            "scale_ratio_nominal_to_current": nominal_scale / current_scale,
            "nominal_current_discrepancy_fraction": nominal_scale / current_scale - 1.0,
            "scale_discrepancy_fraction": nominal_scale / current_scale - 1.0,
            "tissue_hires_scalef": manifest[slide]["tissue_hires_scalef"],
            "tissue_lowres_scalef": manifest[slide]["tissue_lowres_scalef"],
            "pixel_source": pixel_source,
        }
        results.append(result)
    return results


def run_audit(source_h5ad: str | Path, annotated_h5ad: str | Path, output_dir: str | Path,
              *, slide_key: str = "sample_id", domain_columns: str | None = None,
              domain_prefix: str = DEFAULT_DOMAIN_PREFIX,
              sample_manifest: str | Path | None = None) -> Path:
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
        source_slides = _slide_values(source, slide_key, "source")
        slides = _slide_values(annotated, slide_key, "annotated")
        if source_slides != slides:
            mismatches = [
                {"row": index + 1, "obs_id": source_ids[index], "source": source_slides[index], "annotated": slides[index]}
                for index in range(len(source_slides)) if source_slides[index] != slides[index]
            ]
            raise H5ADQCCError(f"source and annotated slide values are not exactly aligned row-by-row; sample={mismatches[:10]}")
        geometry = None
        if sample_manifest is not None:
            manifest_rows = _manifest_geometry(sample_manifest, sorted(set(source_slides)))
            geometry = _source_geometry(source, source_slides, manifest_rows)
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
        output_names = ["zero_count_spots.csv", "invalid_neighborhoods.csv", "zero_count_x_zero_degree_x_validity.csv", "per_slide_summary.csv"]
        if geometry is not None:
            geometry_frame = pd.DataFrame(geometry)
            _atomic_csv(geometry_frame, staging / "source_geometry_summary.csv")
            candidate = geometry_frame[["sample_id", "nominal_100um_scale_um_per_pixel"]].rename(
                columns={"nominal_100um_scale_um_per_pixel": "microns_per_pixel"}
            )
            candidate["scale_source"] = "nominal_100um_array_pitch_sensitivity_candidate"
            _atomic_csv(candidate, staging / "nominal_100um_sensitivity_candidate_scales.csv")
            output_names.extend(["source_geometry_summary.csv", "nominal_100um_sensitivity_candidate_scales.csv"])
        payload = {
            "scope": "read-only full H5AD NOVAE QC cross-tab",
            "source_h5ad": str(source_path.resolve()), "annotated_h5ad": str(annotated_path.resolve()),
            "source_obs": int(source.n_obs), "annotated_obs": int(annotated.n_obs),
            "slide_key": slide_key, "domain_columns": domains,
            "slide_alignment": {"source_annotated_rowwise_exact": True, "source_and_annotated_obs_ids_exact": True},
            "sample_manifest": str(Path(sample_manifest).resolve()) if sample_manifest is not None else None,
            "source_x_contract": "finite_nonnegative_integer_like_raw_counts",
            "zero_count": int(frame["zero_count"].sum()),
            "zero_count_obs_ids": [str(value) for value in zero_spots["obs_id"].tolist()],
            "zero_degree": int(frame["zero_degree"].sum()),
            "invalid_neighborhood": int((~frame["neighborhood_valid"]).sum()),
            "zero_count_filtering_sensitivity": "not justified at this evidence gate; zero-count rows are retained",
            "zero_count_x_zero_degree_x_validity": cross_rows,
            "graph_key": GRAPH_KEY, "neighborhood_valid_key": VALID_KEY,
            "outputs": output_names,
            "per_slide": slide_frame.to_dict(orient="records"),
            "source_geometry": {
                "enabled": geometry is not None,
                "summary": geometry or [],
                "edge_definition": "observed in-tissue source spots with canonical undirected array-coordinate offsets (0,2), (1,-1), (1,1); no nearest-neighbor inference",
                "candidate_file": "nominal_100um_sensitivity_candidate_scales.csv" if geometry is not None else None,
            },
            "caveat": "No rows were filtered and no H5AD input was mutated. Missing domain values are preserved as NA. Geometry is broad evidence from source H5AD metadata, not independent microscope calibration; no calibration correction or sensitivity run is selected or applied. The nominal 100um candidate scales file is non-operational and must never be used as an input manifest.",
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
    parser.add_argument("--sample-manifest", type=Path, help="validated source sample manifest for geometry evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run_audit(args.source_h5ad, args.annotated_h5ad, args.output_dir, slide_key=args.slide_key, domain_columns=args.domain_columns, domain_prefix=args.domain_prefix, sample_manifest=args.sample_manifest)
    except (OSError, H5ADQCCError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
