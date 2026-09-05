#!/usr/bin/env python3
"""Standalone, exploratory NOVAE Phase 0/1 pilot.

This adapter deliberately has no dependency on the project's pipeline runner.  It
owns the Visium pixel-to-micron conversion and writes a separate annotated H5AD.
The public audit/aggregation helpers are usable without importing NOVAE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

NMF_KEYS = ("NMF_factor", "dominant_nmf_factor")
SPATIAL_KEY = "spatial"
ORIGINAL_SPATIAL_KEY = "spatial_original_px"
GRAPH_KEY = "spatial_connectivities"
DISTANCE_KEY = "spatial_distances"
DATASET_ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"
DEFAULT_MODEL_REVISION = "b8c0a5d7612bac6bc719ab57ed3cd16ad814728c"
NEIGHBORHOOD_VALID_KEY = "neighborhood_valid"
NOVAE_NEIGHBORHOOD_VALID_KEY = NEIGHBORHOOD_VALID_KEY
DEFAULT_DOMAIN_ASSIGNMENT_COVERAGE = 0.70
MAX_UNASSIGNED_OBS_IDS = 10


class NovaPilotError(ValueError):
    """An input, provenance, or output contract violation."""


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - exercised on unconfigured machines
        raise RuntimeError("anndata is required to run the NOVAE pilot") from exc
    return ad


def _stringified_mapping_items(value: Mapping[Any, Any], path: str) -> list[tuple[str, Any, Any]]:
    """Return sorted mapping items while rejecting lossy string-key collisions."""
    seen: dict[str, Any] = {}
    items: list[tuple[str, Any, Any]] = []
    for key, item in value.items():
        string_key = str(key)
        if string_key in seen:
            raise NovaPilotError(
                f"{path} has mapping-key collision: {seen[string_key]!r} and {key!r} "
                f"both stringify to {string_key!r}"
            )
        seen[string_key] = key
        items.append((string_key, key, item))
    return sorted(items, key=lambda item: item[0])


def _canonicalize_provenance(value: Any, path: str = "provenance") -> Any:
    """Canonicalize JSON/H5AD-safe values before writing either provenance copy.

    AnnData versions differ in how they serialize ``None`` mapping values.  An
    absent mapping entry is therefore the canonical representation.  Nullable
    sequence members are rejected because they cannot be represented reliably
    by AnnData's list writer; core provenance does not require them. Numeric and
    boolean primitives are deliberately not converted to strings.
    """
    if isinstance(value, Mapping):
        return {
            string_key: _canonicalize_provenance(item, f"{path}.{string_key}")
            for string_key, _, item in _stringified_mapping_items(value, path)
            if item is not None and not (
                isinstance(item, (list, tuple, np.ndarray)) and len(item) == 0
            )
        }
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            if item is None:
                raise NovaPilotError(
                    f"{path}[{index}] is None; nullable sequence members are not supported in provenance"
                )
            result.append(_canonicalize_provenance(item, f"{path}[{index}]"))
        return result
    if isinstance(value, np.ndarray):
        return _canonicalize_provenance(value.tolist(), path)
    if isinstance(value, np.generic):
        return _canonicalize_provenance(value.item(), path)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise NovaPilotError(f"{path} contains unsupported value type: {type(value).__name__}")


def _h5ad_safe_provenance(value: Any, path: str = "provenance") -> Any:
    """Make nested provenance values representable by AnnData's HDF5 writer.

    AnnData writes lists as NumPy arrays, so a list containing mappings (or
    heterogeneous values) becomes an object array and fails during HDF5 string
    conversion.  Record lists are represented as deterministic mappings.  A
    per-slide record list uses the slide identifier as its key and removes the
    now-redundant nested ``slide`` field; other record lists use stable index
    keys so no information is discarded.
    """
    if isinstance(value, Mapping):
        return {
            string_key: _h5ad_safe_provenance(item, f"{path}.{string_key}")
            for string_key, _, item in _stringified_mapping_items(value, path)
            if item is not None and not (
                isinstance(item, (list, tuple, np.ndarray)) and len(item) == 0
            )
        }
    if isinstance(value, np.ndarray):
        return _h5ad_safe_provenance(value.tolist(), path)
    if isinstance(value, np.generic):
        return _h5ad_safe_provenance(value.item(), path)
    if isinstance(value, (list, tuple)):
        items = []
        for index, item in enumerate(value):
            if item is None:
                raise NovaPilotError(
                    f"{path}[{index}] is None; nullable sequence members are not supported in provenance"
                )
            items.append(_h5ad_safe_provenance(item, f"{path}[{index}]"))
        if not items:
            return items
        if all(isinstance(item, Mapping) for item in items):
            if all("slide" in item for item in items):
                keyed: dict[str, Any] = {}
                for item in items:
                    slide = str(item["slide"])
                    if slide in keyed:
                        raise NovaPilotError(f"{path} contains duplicate slide {slide!r}")
                    keyed[slide] = {key: entry for key, entry in item.items() if key != "slide"}
                return {key: keyed[key] for key in sorted(keyed)}
            return {str(index): item for index, item in enumerate(items)}
        scalar_types = {type(item) for item in items}
        numeric_scalars = all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items)
        if all(isinstance(item, (str, bool, int, float)) for item in items) and (
            len(scalar_types) == 1 or numeric_scalars
        ):
            return items
        # This indexed form preserves mixed primitive types and nested values
        # without asking AnnData to coerce them into one NumPy object array.
        return {str(index): item for index, item in enumerate(items)}
    if isinstance(value, (str, int, float, bool)):
        return value
    raise NovaPilotError(f"{path} contains unsupported H5AD provenance type: {type(value).__name__}")


def _assert_h5ad_provenance_serializable(value: Any, path: str = "provenance") -> None:
    """Assert that canonical provenance contains only AnnData-safe structures."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NovaPilotError(f"{path} has a non-string mapping key")
            _assert_h5ad_provenance_serializable(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        if not value or all(isinstance(item, (str, bool, int, float)) for item in value):
            return
        raise NovaPilotError(f"{path} contains a non-scalar list")
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    raise NovaPilotError(f"{path} contains unsupported H5AD provenance type: {type(value).__name__}")


def _canonicalize_core_provenance(value: Any) -> Any:
    """Canonicalize and make embedded core provenance safe for AnnData."""
    safe = _h5ad_safe_provenance(_canonicalize_provenance(value))
    _assert_h5ad_provenance_serializable(safe)
    return safe


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _as_text_values(values: Iterable[Any]) -> list[str]:
    result = []
    for value in values:
        if _is_missing(value):
            raise NovaPilotError("Identifiers and boundary values must not be missing")
        text = str(value)
        if not text:
            raise NovaPilotError("Identifiers and boundary values must not be empty")
        result.append(text)
    return result


def _matrix_audit(matrix: Any) -> dict[str, Any]:
    """Describe expression values without guessing whether they are counts."""
    is_sparse = sparse.issparse(matrix)
    values = np.asarray(matrix.data if is_sparse else matrix)
    shape = tuple(matrix.shape) if is_sparse else values.shape
    if len(shape) != 2:
        raise NovaPilotError("expression matrix must be two-dimensional")
    flat = values.ravel()
    try:
        finite = bool(np.isfinite(flat).all()) if flat.size else True
        nonnegative = bool((flat >= 0).all()) if flat.size else True
        integer_like = bool(np.all(np.isclose(flat, np.rint(flat), rtol=0, atol=1e-8))) if flat.size else True
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("expression matrix must contain numeric values") from exc
    if is_sparse:
        row_totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        row_totals = np.asarray(values).sum(axis=1) if values.ndim == 2 else np.array([], dtype=float)
    zero_count_rows = int(np.count_nonzero(row_totals == 0)) if len(row_totals) else 0
    return {
        "dtype": str(values.dtype),
        "shape": {"rows": int(shape[0]), "columns": int(shape[1])},
        "sparse": bool(is_sparse), "nnz": int(matrix.nnz if is_sparse else np.count_nonzero(matrix)),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "finite": finite, "nonnegative": nonnegative, "integer_like": integer_like,
        "zero_count_rows": zero_count_rows,
    }


def _zero_count_row_audit(matrix: Any, obs_names: Iterable[Any]) -> dict[str, Any]:
    """Record zero-count rows as QC; never silently remove or replace them."""
    values = np.asarray(matrix.sum(axis=1)).ravel() if sparse.issparse(matrix) else np.asarray(matrix).sum(axis=1)
    indices = np.flatnonzero(values == 0)
    names = list(obs_names)
    return {
        "count": int(len(indices)),
        "obs_ids": [str(names[index]) for index in indices[:MAX_UNASSIGNED_OBS_IDS]],
        "sample_truncated": bool(len(indices) > MAX_UNASSIGNED_OBS_IDS),
        "note": "QC only: zero-count rows are retained and require scientific interpretation.",
    }


def _copy_matrix(matrix: Any) -> Any:
    return matrix.copy() if hasattr(matrix, "copy") else np.array(matrix, copy=True)


def _matrices_equal(left: Any, right: Any) -> bool:
    if sparse.issparse(left) and sparse.issparse(right):
        difference = left.tocsr() - right.tocsr()
        return difference.nnz == 0
    if sparse.issparse(left) or sparse.issparse(right):
        left_dense = left.toarray() if sparse.issparse(left) else np.asarray(left)
        right_dense = right.toarray() if sparse.issparse(right) else np.asarray(right)
    else:
        left_dense, right_dense = np.asarray(left), np.asarray(right)
    try:
        return bool(np.array_equal(left_dense, right_dense, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_dense, right_dense))


def expression_audit(adata: Any, expression_mode: str) -> dict[str, Any]:
    """Audit X/layers/raw and enforce the explicit raw-counts contract."""
    if expression_mode not in {"raw_counts", "preprocessed"}:
        raise NovaPilotError(f"unsupported expression mode: {expression_mode!r}")
    audit: dict[str, Any] = {"mode": expression_mode, "X": _matrix_audit(adata.X),
                             "layers": {}, "raw_present": adata.raw is not None,
                             "zero_count_rows": _zero_count_row_audit(adata.X, adata.obs_names)}
    for key in adata.layers.keys():
        audit["layers"][str(key)] = _matrix_audit(adata.layers[key])
    if adata.raw is not None:
        audit["raw"] = _matrix_audit(adata.raw.X)
        audit["raw_shape"] = [int(x) for x in adata.raw.shape]
    else:
        audit["raw"] = None
    x_audit = audit["X"]
    if not x_audit["finite"]:
        raise NovaPilotError("expression X contains non-finite values")
    if expression_mode == "raw_counts" and not (x_audit["nonnegative"] and x_audit["integer_like"]):
        raise NovaPilotError("raw_counts requires finite, nonnegative, integer-like expression X")
    return audit


def post_inference_expression_audit(
    adata: Any, expression_mode: str, original_x: Any,
    auto_preprocessing_enabled: bool, counts_before_present: bool,
) -> dict[str, Any]:
    """Audit NOVAE's post-processing without reapplying the raw-count contract."""
    if expression_mode not in {"raw_counts", "preprocessed"}:
        raise NovaPilotError(f"unsupported expression mode: {expression_mode!r}")
    if expression_mode == "preprocessed" and auto_preprocessing_enabled:
        raise NovaPilotError("preprocessed input requires auto_preprocessing=False")
    current = _matrix_audit(adata.X)
    original_audit = _matrix_audit(original_x)
    if current["shape"] != original_audit["shape"]:
        raise NovaPilotError("post-NOVAE expression X is not row/column aligned to input X")
    if not current["finite"]:
        raise NovaPilotError("post-NOVAE expression X contains non-finite values")
    counts = adata.layers["counts"] if "counts" in adata.layers else None
    counts_audit = _matrix_audit(counts) if counts is not None else None
    counts_equal = False
    if counts is not None and expression_mode == "raw_counts":
        if counts_audit["shape"] != original_audit["shape"]:
            raise NovaPilotError("layers['counts'] is not row/column aligned to input X")
        if not (counts_audit["finite"] and counts_audit["nonnegative"] and counts_audit["integer_like"]):
            raise NovaPilotError("layers['counts'] must preserve finite nonnegative integer-like input counts")
        counts_equal = _matrices_equal(counts, original_x)
        if not counts_equal:
            raise NovaPilotError("layers['counts'] does not equal the original input counts")
    transformed = not _matrices_equal(adata.X, original_x)
    if expression_mode == "raw_counts":
        if not current["nonnegative"]:
            raise NovaPilotError("post-NOVAE raw-count X must remain finite and nonnegative")
        if counts is None and transformed:
            raise NovaPilotError("NOVAE transformed raw-count X without preserving layers['counts']")
    return {
        "mode": expression_mode, "X": current, "counts": counts_audit,
        "zero_count_rows": _zero_count_row_audit(adata.X, adata.obs_names),
        "counts_layer_present": counts is not None,
        "counts_layer_created": bool(counts is not None and not counts_before_present),
        "counts_layer_preserved": bool(counts_equal) if expression_mode == "raw_counts" else None,
        "raw_X_preserved_without_counts": bool(counts is None and not transformed),
        "auto_preprocessing_enabled": bool(auto_preprocessing_enabled),
        "auto_preprocessing_applied": bool(transformed),
    }


def validate_dataset_id(dataset_id: str) -> str:
    import re
    if not re.fullmatch(DATASET_ID_RE, dataset_id):
        raise NovaPilotError("dataset_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return dataset_id


def validate_input(adata: Any, slide_key: str, group_key: str | None = None,
                   spatial_key: str = SPATIAL_KEY) -> dict[str, Any]:
    """Validate stable IDs, boundaries, metadata, and finite Nx2 coordinates."""
    if not adata.obs_names.is_unique:
        raise NovaPilotError("obs names must be unique")
    if not adata.var_names.is_unique:
        raise NovaPilotError("var names must be unique")
    if slide_key not in adata.obs:
        raise NovaPilotError(f"slide key {slide_key!r} is missing")
    slides = _as_text_values(adata.obs[slide_key].tolist())
    slide_group_mapping: dict[str, str] | None = None
    if group_key is not None:
        if group_key not in adata.obs:
            raise NovaPilotError(f"group key {group_key!r} is missing")
        groups = _as_text_values(adata.obs[group_key].tolist())
        slide_group_mapping = {}
        for slide in sorted(set(slides)):
            mapped_groups = sorted({group for row_slide, group in zip(slides, groups) if row_slide == slide})
            if len(mapped_groups) != 1:
                raise NovaPilotError(
                    f"each slide must map to exactly one {group_key!r}; "
                    f"slide {slide!r} maps to {mapped_groups}"
                )
            slide_group_mapping[slide] = mapped_groups[0]
    if spatial_key not in adata.obsm:
        raise NovaPilotError(f"obsm[{spatial_key!r}] is missing")
    try:
        coordinates = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("spatial coordinates must be numeric") from exc
    if coordinates.shape != (adata.n_obs, 2):
        raise NovaPilotError(f"spatial coordinates must have shape ({adata.n_obs}, 2)")
    if not np.isfinite(coordinates).all():
        raise NovaPilotError("spatial coordinates must be finite")
    for slide in sorted(set(slides)):
        rows = coordinates[np.asarray(slides) == slide]
        if len(rows) != len(np.unique(rows, axis=0)):
            raise NovaPilotError(f"duplicate spatial coordinates within slide {slide!r}")
    return {
        "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
        "slides": sorted(set(slides)),
        "slide_counts": {str(key): int(value) for key, value in sorted(pd.Series(slides).value_counts().items())},
        "spatial_shape": list(coordinates.shape),
        "spatial_min": coordinates.min(axis=0).tolist() if len(coordinates) else [],
        "spatial_max": coordinates.max(axis=0).tolist() if len(coordinates) else [],
        **({"slide_group_mapping": slide_group_mapping} if slide_group_mapping is not None else {}),
    }


def _restore_obs_column(adata: Any, key: str, values: pd.Series) -> None:
    """Restore a column even when NOVAE changed its categorical categories."""
    adata.obs[key] = adata.obs[key].astype(object)
    adata.obs[key] = values.copy(deep=True)


def read_sample_manifest(path: str | Path) -> pd.DataFrame:
    """Read a CSV/JSON sample manifest without guessing calibration values."""
    path = Path(path)
    if not path.exists():
        raise NovaPilotError(f"sample manifest does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "samples" in payload or "manifest" in payload:
                payload = payload.get("samples", payload.get("manifest"))
            elif not {"sample_id", "spot_diameter_fullres"}.issubset(payload):
                payload = [{"sample_id": key, "spot_diameter_fullres": value}
                           for key, value in payload.items()]
        frame = pd.DataFrame(payload)
    else:
        frame = pd.read_csv(path)
    required = {"sample_id", "spot_diameter_fullres"}
    missing = required - set(frame.columns)
    if missing:
        raise NovaPilotError(f"sample manifest is missing columns: {sorted(missing)}")
    return frame


def validate_manifest(manifest: pd.DataFrame, observed_slides: Iterable[Any]) -> dict[str, float]:
    """Return one positive full-resolution spot diameter for exactly each slide."""
    if not isinstance(manifest, pd.DataFrame):
        manifest = pd.DataFrame(manifest)
    required = {"sample_id", "spot_diameter_fullres"}
    if required - set(manifest.columns):
        raise NovaPilotError("manifest requires sample_id and spot_diameter_fullres columns")
    observed = set(_as_text_values(observed_slides))
    ids = _as_text_values(manifest["sample_id"].tolist())
    if len(ids) != len(set(ids)):
        duplicates = sorted({x for x in ids if ids.count(x) > 1})
        raise NovaPilotError(f"manifest has duplicate sample_id entries: {duplicates}")
    values: dict[str, float] = {}
    for sample_id, value in zip(ids, manifest["spot_diameter_fullres"].tolist()):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise NovaPilotError(f"invalid spot diameter for {sample_id!r}") from exc
        if not np.isfinite(number) or number <= 0:
            raise NovaPilotError(f"spot diameter for {sample_id!r} must be finite and positive")
        values[sample_id] = number
    if set(values) != observed:
        raise NovaPilotError(
            "manifest slides must exactly match observed slides; "
            f"missing={sorted(observed - set(values))}, extra={sorted(set(values) - observed)}"
        )
    return values


def harmonize_visium_coordinates(
    adata: Any, slide_key: str, manifest: pd.DataFrame | str | Path,
    physical_spot_diameter_um: float, *, source_key: str = SPATIAL_KEY,
    original_key: str = ORIGINAL_SPATIAL_KEY,
) -> pd.DataFrame:
    """Materialize per-slide Visium coordinates in microns, preserving source pixels."""
    try:
        physical = float(physical_spot_diameter_um)
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("physical spot diameter must be numeric") from exc
    if not np.isfinite(physical) or physical <= 0:
        raise NovaPilotError("physical spot diameter must be finite and positive")
    validate_input(adata, slide_key, spatial_key=source_key)
    frame = read_sample_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
    scales = validate_manifest(frame, adata.obs[slide_key].tolist())
    pixels = np.asarray(adata.obsm[source_key], dtype=np.float64)
    # Copy before changing spatial: source order and exact pixel values remain auditable.
    adata.obsm[original_key] = pixels.copy()
    slides = np.asarray([str(x) for x in adata.obs[slide_key].tolist()])
    microns = pixels.copy()
    rows: list[dict[str, Any]] = []
    for slide in sorted(scales):
        factor = physical / scales[slide]
        mask = slides == slide
        microns[mask] *= factor
        rows.append({
            "slide": slide, "spot_diameter_fullres_px": scales[slide],
            "physical_spot_diameter_um": physical,
            "microns_per_pixel": factor, "n_obs": int(mask.sum()),
            "original_x_min_px": float(pixels[mask, 0].min()),
            "original_x_max_px": float(pixels[mask, 0].max()),
            "original_y_min_px": float(pixels[mask, 1].min()),
            "original_y_max_px": float(pixels[mask, 1].max()),
            "harmonized_x_min_um": float(microns[mask, 0].min()),
            "harmonized_x_max_um": float(microns[mask, 0].max()),
            "harmonized_y_min_um": float(microns[mask, 1].min()),
            "harmonized_y_max_um": float(microns[mask, 1].max()),
        })
    adata.obsm[source_key] = microns
    return pd.DataFrame(rows)


def _nmf_snapshot(adata: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in NMF_KEYS:
        if key in adata.obs:
            series = adata.obs[key]
            snapshot[key] = {
                "values": series.astype(object).tolist(),
                "dtype": str(series.dtype),
                "categories": series.cat.categories.astype(str).tolist() if hasattr(series.dtype, "categories") else None,
            }
    return snapshot


def _values_equal_with_missing(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    for old, new in zip(left, right):
        if _is_missing(old) or _is_missing(new):
            if not (_is_missing(old) and _is_missing(new)):
                return False
        elif old != new:
            return False
    return True


def assert_nmf_unchanged(adata: Any, snapshot: Mapping[str, Any]) -> None:
    current = _nmf_snapshot(adata)
    if set(current) != set(snapshot):
        raise NovaPilotError("NOVAE adapter changed the set of NMF columns")
    for key in snapshot:
        old, new = snapshot[key], current[key]
        if (not _values_equal_with_missing(old["values"], new["values"])
                or old["dtype"] != new["dtype"] or old["categories"] != new["categories"]):
            raise NovaPilotError(f"NOVAE adapter changed existing NMF column {key!r}")


def validate_spatial_graph(adata: Any, slide_key: str, graph_key: str = GRAPH_KEY) -> dict[str, int]:
    """Validate graph dimensions and the hard slide partition."""
    if graph_key not in adata.obsp:
        raise NovaPilotError(f"obsp[{graph_key!r}] is missing")
    graph = adata.obsp[graph_key]
    if not sparse.issparse(graph):
        raise NovaPilotError("spatial graph must be a scipy sparse matrix")
    if graph.shape != (adata.n_obs, adata.n_obs):
        raise NovaPilotError("spatial graph shape does not match observations")
    graph = graph.tocsr().copy()
    graph.eliminate_zeros()
    slides = np.asarray([str(x) for x in adata.obs[slide_key].tolist()])
    rows, cols = graph.nonzero()
    cross = int(np.count_nonzero(slides[rows] != slides[cols]))
    if cross:
        raise NovaPilotError(f"spatial graph has {cross} cross-slide edges")
    return {"n_nonzero_directed": int(graph.nnz), "cross_slide_edges": cross}


def prune_spatial_graph_radius(
    adata: Any, slide_key: str, radius_um: float, effective_scale: float,
    graph_key: str = GRAPH_KEY, distance_key: str = DISTANCE_KEY,
) -> dict[str, Any]:
    """Prune an existing graph by physical coordinate distance, preserving topology."""
    try:
        radius = float(radius_um)
        scale = float(effective_scale)
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("graph radius and effective scale must be numeric") from exc
    if not np.isfinite(radius) or radius <= 0:
        raise NovaPilotError("graph radius must be finite and positive")
    if not np.isfinite(scale) or scale <= 0:
        raise NovaPilotError("effective coordinate scale must be finite and positive")
    before = validate_spatial_graph(adata, slide_key, graph_key)
    graph = adata.obsp[graph_key].tocsr().copy()
    graph.eliminate_zeros()
    distances = adata.obsp.get(distance_key)
    if distances is None:
        raise NovaPilotError(f"obsp[{distance_key!r}] is missing; corresponding distances are required")
    if not sparse.issparse(distances):
        distances = sparse.csr_matrix(distances)
    if distances.shape != graph.shape:
        raise NovaPilotError("spatial distances shape does not match observations")
    try:
        coordinates = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise NovaPilotError("spatial coordinates are required for graph radius pruning") from exc
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise NovaPilotError("spatial coordinates must be finite and row-aligned for graph radius pruning")
    coo = graph.tocoo()
    physical_distances = np.sqrt(np.sum((coordinates[coo.row] - coordinates[coo.col]) ** 2, axis=1)) * scale
    keep = physical_distances <= radius
    pruned = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape)
    pruned.eliminate_zeros()
    keep_mask = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (coo.row[keep], coo.col[keep])), shape=graph.shape
    )
    pruned_distances = distances.tocsr().multiply(keep_mask).tocsr()
    slides = np.asarray([str(value) for value in adata.obs[slide_key].tolist()])
    per_slide: list[dict[str, Any]] = []
    for slide in sorted(set(slides)):
        indices = np.flatnonzero(slides == slide)
        before_sub = graph[indices][:, indices]
        after_sub = pruned[indices][:, indices]
        before_undirected = before_sub.maximum(before_sub.T).tocsr()
        before_undirected.eliminate_zeros()
        after_undirected = after_sub.maximum(after_sub.T).tocsr()
        after_undirected.eliminate_zeros()
        before_edges = int(len(_edge_pairs(sparse.triu(before_undirected, k=1))[0]))
        after_edges = int(len(_edge_pairs(sparse.triu(after_undirected, k=1))[0]))
        if len(indices) > 1 and before_edges > 0 and after_edges == 0:
            raise NovaPilotError(
                f"graph radius pruning destroyed all edges in nontrivial slide {slide!r}"
            )
        per_slide.append({
            "slide": slide, "obs": int(len(indices)),
            "pre_directed_edges": int(before_sub.nnz), "post_directed_edges": int(after_sub.nnz),
            "removed_directed_edges": int(before_sub.nnz - after_sub.nnz),
            "pre_undirected_edges": before_edges, "post_undirected_edges": after_edges,
            "removed_undirected_edges": int(before_edges - after_edges),
            "pre_zero_degree_count": int(np.count_nonzero(np.asarray(before_undirected.getnnz(axis=1)).ravel() == 0)),
            "post_zero_degree_count": int(np.count_nonzero(np.asarray(after_undirected.getnnz(axis=1)).ravel() == 0)),
        })
    adata.obsp[graph_key] = pruned
    adata.obsp[distance_key] = pruned_distances
    after = validate_spatial_graph(adata, slide_key, graph_key)
    pre_undirected_edges = sum(row["pre_undirected_edges"] for row in per_slide)
    post_undirected_edges = sum(row["post_undirected_edges"] for row in per_slide)
    return {
        "requested_radius_um": radius, "effective_scale_to_microns": scale,
        "distance_basis": "euclidean_obsm_spatial_edge_distance_times_effective_scale",
        "applied": True, "pre": before, "post": after,
        "pre_directed_edges": int(before["n_nonzero_directed"]),
        "post_directed_edges": int(after["n_nonzero_directed"]),
        "removed_directed_edges": int(before["n_nonzero_directed"] - after["n_nonzero_directed"]),
        "pre_undirected_edges": int(pre_undirected_edges),
        "post_undirected_edges": int(post_undirected_edges),
        "removed_undirected_edges": int(pre_undirected_edges - post_undirected_edges),
        "pre_zero_degree_count": int(sum(row["pre_zero_degree_count"] for row in per_slide)),
        "post_zero_degree_count": int(sum(row["post_zero_degree_count"] for row in per_slide)),
        "per_slide": per_slide,
    }


def _edge_pairs(graph: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    coo = graph.tocoo()
    mask = coo.row < coo.col
    return coo.row[mask], coo.col[mask]


def graph_diagnostics(adata: Any, slide_key: str, graph_key: str = GRAPH_KEY,
                      distance_key: str = DISTANCE_KEY) -> pd.DataFrame:
    """Return per-slide directed/undirected, degree, component, and distance metrics."""
    validate_spatial_graph(adata, slide_key, graph_key)
    graph = adata.obsp[graph_key].tocsr().copy()
    graph.eliminate_zeros()
    distances = adata.obsp.get(distance_key)
    if distances is not None and not sparse.issparse(distances):
        distances = sparse.csr_matrix(distances)
    slides = np.asarray([str(x) for x in adata.obs[slide_key].tolist()])
    rows: list[dict[str, Any]] = []
    for slide in sorted(set(slides)):
        indices = np.flatnonzero(slides == slide)
        sub = graph[indices][:, indices]
        undirected = sub.maximum(sub.T).tocsr()
        undirected.eliminate_zeros()
        pair_rows, pair_cols = _edge_pairs(undirected)
        degrees = np.asarray(undirected.getnnz(axis=1)).ravel()
        record: dict[str, Any] = {
            "slide": slide, "obs": int(len(indices)),
            "directed_edges": int(sub.nnz), "undirected_edges": int(len(pair_rows)),
            "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
            "min_degree": int(degrees.min()) if len(degrees) else 0,
            "max_degree": int(degrees.max()) if len(degrees) else 0,
            "zero_degree_count": int(np.count_nonzero(degrees == 0)),
            "connected_components": int(connected_components(sub, directed=False, return_labels=False)) if len(indices) else 0,
            "cross_slide_violation_count": 0,
            "distance_count": 0, "distance_min": np.nan,
            "distance_mean": np.nan, "distance_max": np.nan,
        }
        if distances is not None and distances.shape == graph.shape:
            dsub = distances[indices][:, indices]
            vals = np.asarray(dsub.data, dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            record.update({
                "distance_count": int(len(vals)),
                "distance_min": float(vals.min()) if len(vals) else np.nan,
                "distance_mean": float(vals.mean()) if len(vals) else np.nan,
                "distance_max": float(vals.max()) if len(vals) else np.nan,
            })
        rows.append(record)
    return pd.DataFrame(rows)


def _domain_assigned_mask(values: Iterable[Any]) -> np.ndarray:
    """Identify real labels without converting missing values to the string ``nan``."""
    mask: list[bool] = []
    for value in values:
        if _is_missing(value):
            mask.append(False)
            continue
        text = str(value).strip()
        mask.append(bool(text) and text.lower() != "nan")
    return np.asarray(mask, dtype=bool)


def _neighborhood_valid_mask(adata: Any, key: str = NEIGHBORHOOD_VALID_KEY) -> np.ndarray:
    if key not in adata.obs:
        raise NovaPilotError(f"NOVAE did not produce expected {key!r} column")
    values = adata.obs[key].tolist()
    mask: list[bool] = []
    for value in values:
        if _is_missing(value) or not isinstance(value, (bool, np.bool_)):
            raise NovaPilotError(f"{key!r} must contain only non-missing boolean values")
        mask.append(bool(value))
    return np.asarray(mask, dtype=bool)


def audit_domain_assignments(
    adata: Any, slide_key: str, domain_key: str,
    minimum_coverage: float = DEFAULT_DOMAIN_ASSIGNMENT_COVERAGE,
    *, neighborhood_key: str = NEIGHBORHOOD_VALID_KEY,
) -> dict[str, Any]:
    """Audit NOVAE labels against its explicit valid-neighborhood mask."""
    if slide_key not in adata.obs:
        raise NovaPilotError(f"grouping key {slide_key!r} is missing")
    if domain_key not in adata.obs:
        raise NovaPilotError(f"NOVAE did not produce expected domain key {domain_key!r}")
    try:
        threshold = float(minimum_coverage)
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("minimum domain-assignment coverage must be numeric") from exc
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise NovaPilotError("minimum domain-assignment coverage must be in [0, 1]")
    valid = _neighborhood_valid_mask(adata, neighborhood_key)
    domains = adata.obs[domain_key].tolist()
    assigned = _domain_assigned_mask(domains)
    # A valid neighborhood must always have a genuine domain. Invalid rows are
    # expected to retain NOVAE's missing value (and are never imputed).
    invalid_valid = np.flatnonzero(valid & ~assigned)
    invalid_domain = np.flatnonzero(~valid & assigned)
    if len(invalid_valid):
        ids = [str(adata.obs_names[index]) for index in invalid_valid[:MAX_UNASSIGNED_OBS_IDS]]
        raise NovaPilotError(
            f"domain key {domain_key!r} is missing for {len(invalid_valid)} valid neighborhoods; "
            f"sample obs IDs={ids}"
        )
    if len(invalid_domain):
        ids = [str(adata.obs_names[index]) for index in invalid_domain[:MAX_UNASSIGNED_OBS_IDS]]
        raise NovaPilotError(
            f"domain key {domain_key!r} assigns {len(invalid_domain)} invalid neighborhoods; "
            f"sample obs IDs={ids}"
        )
    slides = adata.obs[slide_key].astype(str).to_numpy()
    def counts(indices: np.ndarray) -> dict[str, Any]:
        total = int(len(indices))
        valid_count = int(valid[indices].sum())
        assigned_count = int(assigned[indices].sum())
        unassigned_count = total - assigned_count
        coverage = assigned_count / total if total else 0.0
        unassigned_indices = indices[~assigned[indices]]
        return {
            "total": total, "valid_neighborhood": valid_count,
            "assigned": assigned_count, "unassigned": unassigned_count,
            "coverage": coverage,
            "total_observations": total, "valid_neighborhood_observations": valid_count,
            "assigned_observations": assigned_count, "unassigned_observations": unassigned_count,
            "assignment_coverage": coverage,
            "unassigned_obs_ids": [str(adata.obs_names[index]) for index in unassigned_indices[:MAX_UNASSIGNED_OBS_IDS]],
            "unassigned_obs_ids_truncated": bool(len(unassigned_indices) > MAX_UNASSIGNED_OBS_IDS),
            "meets_minimum_coverage": bool(coverage >= threshold),
        }
    overall = counts(np.arange(adata.n_obs))
    per_slide = {slide: counts(np.flatnonzero(slides == slide)) for slide in sorted(set(slides))}
    failing_slides = {
        slide: values for slide, values in per_slide.items()
        if values["coverage"] < threshold
    }
    if failing_slides or overall["coverage"] < threshold:
        slide_details = "; ".join(
            f"{slide}: total={values['total']}, valid_neighborhood={values['valid_neighborhood']}, "
            f"assigned={values['assigned']}, unassigned={values['unassigned']}, "
            f"coverage={values['coverage']:.4f}, minimum={threshold:.4f}, "
            f"sample obs IDs={values['unassigned_obs_ids']}"
            for slide, values in failing_slides.items()
        ) or "none"
        overall_detail = (
            f"total={overall['total']}, valid_neighborhood={overall['valid_neighborhood']}, "
            f"assigned={overall['assigned']}, unassigned={overall['unassigned']}, "
            f"coverage={overall['coverage']:.4f}, minimum={threshold:.4f}, "
            f"sample obs IDs={overall['unassigned_obs_ids']}"
        )
        raise NovaPilotError(
            f"domain assignment coverage below minimum for {domain_key!r}; "
            f"failing slides: {slide_details}; overall: {overall_detail}"
        )
    return {"domain_key": domain_key, "neighborhood_key": neighborhood_key,
            "minimum_coverage": threshold,
            "minimum_domain_assignment_coverage": threshold,
            "overall": overall, "per_slide": per_slide}


def domain_adjacency(adata: Any, slide_key: str, domain_key: str,
                     graph_key: str = GRAPH_KEY) -> pd.DataFrame:
    """Count labeled undirected edges, excluding edges touching unassigned rows."""
    if domain_key not in adata.obs:
        raise NovaPilotError(f"domain key {domain_key!r} is missing")
    validate_spatial_graph(adata, slide_key, graph_key)
    graph = adata.obsp[graph_key].tocsr().copy()
    graph.eliminate_zeros()
    slides = np.asarray([str(x) for x in adata.obs[slide_key].tolist()])
    domains = adata.obs[domain_key].tolist()
    assigned = _domain_assigned_mask(domains)
    rows, cols = _edge_pairs(sparse.triu(graph.maximum(graph.T), k=1))
    totals = {slide: 0 for slide in sorted(set(slides))}
    used = {slide: 0 for slide in totals}
    counts: dict[tuple[str, str, str], int] = {}
    for left, right in zip(rows, cols):
        if slides[left] != slides[right]:
            raise NovaPilotError("cross-slide edge encountered while building adjacency")
        slide = slides[left]
        totals[slide] += 1
        if not (assigned[left] and assigned[right]):
            continue
        used[slide] += 1
        a, b = sorted((str(domains[left]).strip(), str(domains[right]).strip()))
        key = (slide, a, b)
        counts[key] = counts.get(key, 0) + 1
    records: list[dict[str, Any]] = []
    for slide in sorted(totals):
        slide_counts = [(key, count) for key, count in counts.items() if key[0] == slide]
        if not slide_counts:
            slide_counts = [((slide, "", ""), 0)]
        for (_, a, b), count in sorted(slide_counts):
            total_edges, used_edges = totals[slide], used[slide]
            records.append({
                "slide": slide, "domain_a": a, "domain_b": b, "edge_count": count,
                "edge_proportion": count / used_edges if used_edges else 0.0,
                "proportion": count / used_edges if used_edges else 0.0,
                "total_edges": total_edges, "used_edges": used_edges,
                "excluded_edges": total_edges - used_edges,
                "edge_coverage": used_edges / total_edges if total_edges else 0.0,
                "total": total_edges, "used": used_edges,
                "excluded": total_edges - used_edges, "coverage": used_edges / total_edges if total_edges else 0.0,
            })
    return pd.DataFrame(records)


def domain_proportions(adata: Any, unit_key: str, domain_key: str) -> pd.DataFrame:
    if unit_key not in adata.obs:
        raise NovaPilotError(f"grouping key {unit_key!r} is missing")
    if domain_key not in adata.obs:
        raise NovaPilotError(f"domain key {domain_key!r} is missing")
    units = adata.obs[unit_key].astype(str).to_numpy()
    domains = adata.obs[domain_key].tolist()
    assigned = _domain_assigned_mask(domains)
    totals = pd.Series(units).value_counts().to_dict()
    assigned_counts = pd.Series(units[assigned]).value_counts().to_dict()
    records: list[dict[str, Any]] = []
    for unit in sorted(totals):
        total = int(totals[unit]); assigned_count = int(assigned_counts.get(unit, 0))
        audit = {"total": total, "assigned": assigned_count, "unassigned": total - assigned_count,
                 "coverage": assigned_count / total if total else 0.0}
        labels = pd.Series([str(domains[index]).strip() for index in np.flatnonzero((units == unit) & assigned)])
        counts = labels.value_counts(sort=True)
        for domain, count in counts.items():
            records.append({"domain": domain, "obs": int(count),
                            "proportion": count / assigned_count if assigned_count else 0.0,
                            "total": audit["total"], "assigned": audit["assigned"],
                            "unassigned": audit["unassigned"], "coverage": audit["coverage"],
                            "total_observations": audit["total"], "assigned_observations": audit["assigned"],
                            "unassigned_observations": audit["unassigned"], "assignment_coverage": audit["coverage"],
                            unit_key: unit})
        if not len(counts):
            records.append({"domain": "", "obs": 0, "proportion": 0.0,
                            "total": audit["total"], "assigned": 0,
                            "unassigned": audit["unassigned"], "coverage": audit["coverage"],
                            "total_observations": audit["total"], "assigned_observations": 0,
                            "unassigned_observations": audit["unassigned"], "assignment_coverage": audit["coverage"],
                            unit_key: unit})
    return pd.DataFrame(records)


def latent_summary(adata: Any, slide_key: str, latent_key: str,
                   group_key: str | None = None) -> pd.DataFrame:
    """Summarize finite latents over valid NOVAE neighborhoods only."""
    if latent_key not in adata.obsm:
        raise NovaPilotError(f"latent key {latent_key!r} is missing")
    latent = np.asarray(adata.obsm[latent_key], dtype=float)
    if latent.ndim != 2 or latent.shape[0] != adata.n_obs or not np.isfinite(latent).all():
        raise NovaPilotError("latent representation must be finite, 2-D, and row-aligned")
    valid = _neighborhood_valid_mask(adata)
    keys = [slide_key] + ([group_key] if group_key else [])
    labels = pd.DataFrame({key: adata.obs[key].astype(str).to_numpy() for key in keys})
    labels["_row"] = np.arange(adata.n_obs)
    output: list[dict[str, Any]] = []
    for values, index in labels.groupby(keys, sort=True, observed=True):
        if not isinstance(values, tuple):
            values = (values,)
        all_unit_indices = index["_row"].to_numpy()
        unit_indices = all_unit_indices[valid[all_unit_indices]]
        matrix = latent[unit_indices]
        record = {key: value for key, value in zip(keys, values)}
        record.update({"obs": int(len(matrix)), "total": int(len(all_unit_indices)),
                       "valid": int(len(unit_indices)), "excluded": int(len(all_unit_indices) - len(unit_indices)),
                       "total_observations": int(len(all_unit_indices)),
                       "valid_neighborhood_observations": int(len(unit_indices)),
                       "excluded_observations": int(len(all_unit_indices) - len(unit_indices))})
        for dimension in range(latent.shape[1]):
            record[f"latent_{dimension}_mean"] = float(matrix[:, dimension].mean()) if len(matrix) else np.nan
            record[f"latent_{dimension}_std"] = float(matrix[:, dimension].std(ddof=0)) if len(matrix) else np.nan
        output.append(record)
    return pd.DataFrame(output)


def normalize_resolutions(values: Iterable[Any], primary: Any = 1.0) -> tuple[list[float], float]:
    """Validate a label-free resolution sweep and its predeclared primary."""
    try:
        resolutions = [float(value) for value in values]
        primary_value = float(primary)
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("resolutions and primary resolution must be numeric") from exc
    if not resolutions or not np.isfinite(resolutions).all() or any(value <= 0 for value in resolutions):
        raise NovaPilotError("resolutions must be finite and positive")
    if len(set(resolutions)) != len(resolutions):
        raise NovaPilotError("resolutions must be unique")
    if not np.isfinite(primary_value) or primary_value <= 0:
        raise NovaPilotError("primary resolution must be finite and positive")
    matched = [value for value in resolutions if np.isclose(primary_value, value, rtol=0, atol=1e-12)]
    if not matched:
        raise NovaPilotError("primary resolution must be included in resolutions")
    return resolutions, matched[0]


def resolution_token(resolution: float) -> str:
    """Return a filesystem-safe, stable token for a resolution."""
    text = format(float(resolution), ".12g")
    return text.replace("-", "m").replace(".", "p")


def neighbor_distance_calibration(
    adata: Any, slide_key: str, expected_um: float, relative_tolerance: float = 0.5,
    graph_key: str = GRAPH_KEY, distance_key: str = DISTANCE_KEY,
) -> pd.DataFrame:
    """Check median positive physical graph-edge distances per slide.

    Distances are read from materialized ``obsp[distance_key]`` and only one
    undirected edge is counted.  A Visium lattice can contain diagonal edges or
    tissue gaps, hence this is deliberately a median/tolerance check rather than
    an assertion that every edge equals the nominal center spacing.
    """
    try:
        expected = float(expected_um)
        tolerance = float(relative_tolerance)
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("expected neighbor distance and tolerance must be numeric") from exc
    if not np.isfinite(expected) or expected <= 0:
        raise NovaPilotError("expected neighbor distance must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0 or tolerance >= 1:
        raise NovaPilotError("relative distance tolerance must be finite and in [0, 1)")
    validate_spatial_graph(adata, slide_key, graph_key)
    distances = adata.obsp.get(distance_key)
    if distances is None:
        raise NovaPilotError(f"obsp[{distance_key!r}] is missing; calibrated distances are required")
    if not sparse.issparse(distances):
        distances = sparse.csr_matrix(distances)
    if distances.shape != (adata.n_obs, adata.n_obs):
        raise NovaPilotError("spatial distances shape does not match observations")
    distances = distances.tocsr().maximum(distances.T).tocsr()
    graph = adata.obsp[graph_key].tocsr().maximum(adata.obsp[graph_key].T).tocsr()
    graph.eliminate_zeros()
    slides = np.asarray([str(x) for x in adata.obs[slide_key].tolist()])
    lower, upper = expected * (1 - tolerance), expected * (1 + tolerance)
    records: list[dict[str, Any]] = []
    for slide in sorted(set(slides)):
        indices = np.flatnonzero(slides == slide)
        sub = graph[indices][:, indices]
        rows, cols = _edge_pairs(sub)
        dsub = distances[indices][:, indices]
        values = np.asarray(dsub[rows, cols]).ravel() if len(rows) else np.array([], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        median = float(np.median(values)) if len(values) else np.nan
        passed = bool(len(values) and lower <= median <= upper)
        records.append({
            "slide": slide, "expected_neighbor_distance_um": expected,
            "relative_tolerance": tolerance, "lower_bound_um": lower, "upper_bound_um": upper,
            "positive_undirected_edge_count": int(len(values)),
            "median_neighbor_distance_um": median, "pass": passed,
        })
    result = pd.DataFrame(records)
    failed = result.loc[~result["pass"], "slide"].tolist()
    if failed:
        raise NovaPilotError(
            "Visium neighbor-distance calibration outside declared tolerance for slides: "
            + ", ".join(map(str, failed))
        )
    return result


def _science_metrics(novae: Any, adata: Any, slide_key: str, domain_key: str) -> dict[str, float]:
    """Call the authoritative v1.1.1 NOVAE metric API, without substitutes."""
    try:
        from novae import monitor
    except (ImportError, AttributeError) as exc:
        raise NovaPilotError("NOVAE monitor metric API is unavailable") from exc
    fide = getattr(monitor, "mean_fide_score", None)
    jsd = getattr(monitor, "jensen_shannon_divergence", None)
    if not callable(fide) or not callable(jsd):
        raise NovaPilotError(
            "installed NOVAE does not expose v1.1.1 monitor.mean_fide_score and "
            "monitor.jensen_shannon_divergence; refusing a substitute metric"
        )
    try:
        result = {
            "FIDE": float(fide(adata, obs_key=domain_key, slide_key=slide_key)),
            "JSD": float(jsd(adata, obs_key=domain_key, slide_key=slide_key)),
        }
    except Exception as exc:
        raise NovaPilotError(f"NOVAE official metric API failed for {domain_key!r}: {exc}") from exc
    nonfinite = [name for name, value in result.items() if not np.isfinite(value)]
    if nonfinite:
        raise NovaPilotError(
            f"NOVAE official FIDE/JSD returned non-finite {', '.join(nonfinite)} "
            f"for {domain_key!r}: {result}"
        )
    return result


def _domain_resolution_summary(
    adata: Any, slide_key: str, resolutions: list[float],
    domain_keys: Mapping[float, str], primary_resolution: float,
    assignment_audits: Mapping[float, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slides = adata.obs[slide_key].astype(str).to_numpy()
    for resolution in resolutions:
        key = domain_keys[resolution]
        raw_values = adata.obs[key].tolist()
        assigned_mask = _domain_assigned_mask(raw_values)
        values = np.asarray([str(value).strip() if assigned_mask[index] else "" for index, value in enumerate(raw_values)])
        assigned_values = values[assigned_mask]
        sizes = pd.Series(assigned_values).value_counts(sort=False)
        audit = assignment_audits[resolution] if assignment_audits else None
        overall = audit["overall"] if audit else {
            "total": int(adata.n_obs), "valid_neighborhood": int(assigned_mask.sum()),
            "assigned": int(assigned_mask.sum()), "unassigned": int((~assigned_mask).sum()),
            "coverage": float(assigned_mask.mean()) if len(assigned_mask) else 0.0,
        }
        per_slide = []
        for slide in sorted(set(slides)):
            slide_mask = (slides == slide) & assigned_mask
            per_slide.append(f"{slide}:{int(pd.Series(values[slide_mask]).nunique())}")
        rows.append({
            "resolution": resolution, "primary": bool(np.isclose(resolution, primary_resolution, rtol=0, atol=1e-12)),
            "domain_key": key, "number_of_domains": int(pd.Series(assigned_values).nunique()),
            "total_observations": overall["total"], "valid_neighborhood_observations": overall["valid_neighborhood"],
            "assigned_observations": overall["assigned"], "unassigned_observations": overall["unassigned"],
            "assignment_coverage": overall["coverage"],
            "total": overall["total"], "valid_neighborhood": overall["valid_neighborhood"],
            "assigned": overall["assigned"], "unassigned": overall["unassigned"], "coverage": overall["coverage"],
            "minimum_domain_size": int(sizes.min()) if len(sizes) else 0,
            "median_domain_size": float(sizes.median()) if len(sizes) else 0.0,
            "maximum_domain_size": int(sizes.max()) if len(sizes) else 0,
            "domains_present_per_slide": ";".join(per_slide),
            "unassigned_obs_ids": ";".join(overall.get("unassigned_obs_ids", [])),
            "assignment_audit_per_slide": json.dumps(audit.get("per_slide", {}) if audit else {}, sort_keys=True),
            "per_slide_total": json.dumps({slide: values["total"] for slide, values in (audit.get("per_slide", {}) if audit else {}).items()}, sort_keys=True),
            "per_slide_valid_neighborhood": json.dumps({slide: values["valid_neighborhood"] for slide, values in (audit.get("per_slide", {}) if audit else {}).items()}, sort_keys=True),
            "per_slide_assigned": json.dumps({slide: values["assigned"] for slide, values in (audit.get("per_slide", {}) if audit else {}).items()}, sort_keys=True),
            "per_slide_unassigned": json.dumps({slide: values["unassigned"] for slide, values in (audit.get("per_slide", {}) if audit else {}).items()}, sort_keys=True),
            "per_slide_coverage": json.dumps({slide: values["coverage"] for slide, values in (audit.get("per_slide", {}) if audit else {}).items()}, sort_keys=True),
        })
    return pd.DataFrame(rows)


def _validate_domain_column(adata: Any, domain_key: str) -> None:
    if domain_key not in adata.obs:
        raise NovaPilotError(f"NOVAE did not produce expected domain key {domain_key!r}")
    # Detailed validity/coverage checks are performed after each assignment.


def _validate_inference_outputs(adata: Any, domain_key: str) -> int:
    if domain_key not in adata.obs:
        raise NovaPilotError(f"NOVAE did not produce expected domain key {domain_key!r}")
    _neighborhood_valid_mask(adata)
    candidates = [key for key in adata.obsm.keys() if key == "novae_latent" or key.startswith("novae_latent")]
    if not candidates:
        raise NovaPilotError("NOVAE did not produce a latent representation")
    latent_key = candidates[0]
    latent = np.asarray(adata.obsm[latent_key])
    if latent.ndim != 2 or latent.shape[0] != adata.n_obs or not np.isfinite(latent).all():
        raise NovaPilotError("NOVAE latent representation is not finite and row-aligned")
    return int(latent.shape[1])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> dict[str, str]:
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {str(p.relative_to(root)): sha256_file(p) for p in files}


def _artifact_sha256(path: Path) -> str:
    if path.is_dir():
        return hashlib.sha256(json.dumps(sha256_directory(path), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return sha256_file(path)


def _unique_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False)
    handle.close()
    return Path(handle.name)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _unique_sibling(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = _unique_sibling(path)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_h5ad(adata: Any, path: Path) -> None:
    temporary = _unique_sibling(path)
    try:
        adata.write_h5ad(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_model(model_source: str, model_revision: str | None) -> tuple[Path, dict[str, Any]]:
    source = Path(model_source).expanduser()
    if source.exists():
        if not source.is_dir():
            raise NovaPilotError("local model source must be a directory")
        return source, {"source": "local", "requested_revision": model_revision,
                        "resolved_revision": None, "resolved_revision_status": "not_independently_resolved",
                        "revision_verified": False,
                        "revision_verification": "local_directory_revision_not_independently_resolved",
                        "files_sha256": sha256_directory(source)}
    # Resolve a commit and materialize a snapshot. This intentionally fails closed
    # when a remote cannot be pinned, rather than claiming reproducibility.
    try:
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().model_info(model_source, revision=model_revision)
        resolved = getattr(info, "sha", None)
        if not resolved:
            raise NovaPilotError(f"Hugging Face model {model_source!r} has no resolved commit")
        # Use Hugging Face's normal/configurable cache (HF_HOME/HF_HUB_CACHE),
        # never a path inside the output transaction.
        snapshot = Path(snapshot_download(model_source, revision=resolved))
    except Exception as exc:
        raise NovaPilotError(
            f"could not resolve and pin remote model {model_source!r}; use a local cached model path ({exc})"
        ) from exc
    return snapshot, {"source": "huggingface", "repo_id": model_source,
                      "requested_revision": model_revision, "resolved_revision": resolved,
                      "resolved_revision_status": "resolved_commit_sha",
                      "revision_verified": True,
                      "revision_verification": "huggingface_model_info_commit",
                      "files_sha256": sha256_directory(snapshot)}


def _add_checkpoint_hash(model_provenance: dict[str, Any]) -> dict[str, Any]:
    """Add a deterministic digest over the complete resolved model snapshot."""
    hashes = model_provenance.get("files_sha256", {})
    model_provenance["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return model_provenance


def _configure_scale(novae: Any, scale: float) -> None:
    settings = getattr(novae, "settings", None)
    if settings is None or not hasattr(settings, "scale_to_microns"):
        raise NovaPilotError("installed NOVAE does not expose settings.scale_to_microns")
    settings.scale_to_microns = float(scale)


def _configure_preprocessing(novae: Any, expression_mode: str) -> bool:
    settings = getattr(novae, "settings", None)
    if settings is None or not hasattr(settings, "auto_preprocessing"):
        raise NovaPilotError("installed NOVAE does not expose settings.auto_preprocessing")
    enabled = expression_mode == "raw_counts"
    settings.auto_preprocessing = enabled
    return bool(settings.auto_preprocessing)


def _runtime_device(accelerator: str) -> str:
    try:
        import torch
        if accelerator in {"gpu", "cuda", "auto"} and torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    except ImportError:
        pass
    return accelerator


def _runtime_provenance(novae_version: str | None = None) -> dict[str, Any]:
    """Capture resolved package and accelerator details without requiring CUDA."""
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python >=3.11 always has this
        metadata = None
    distributions = {
        "novae": novae_version, "anndata": "anndata", "numpy": "numpy",
        "pandas": "pandas", "scipy": "scipy", "h5py": "h5py",
        "torch": "torch", "torch-geometric": "torch-geometric",
        "lightning": "lightning", "scanpy": "scanpy",
        "huggingface-hub": "huggingface-hub",
    }
    packages: dict[str, Any] = {}
    for name, distribution in distributions.items():
        if name == "novae" and novae_version is not None:
            packages[name] = str(novae_version)
            continue
        if metadata is not None:
            try:
                packages[name] = metadata.version(str(distribution))
            except metadata.PackageNotFoundError:
                continue
    details: dict[str, Any] = {"packages": packages}
    try:
        import torch
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "compiled_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
        }
        if cuda["available"]:
            index = int(torch.cuda.current_device())
            cuda["current_device"] = index
            cuda["device_name"] = str(torch.cuda.get_device_name(index))
        details["torch"] = {"version": str(torch.__version__), "cuda": cuda}
    except ImportError:
        pass
    return details


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Exploratory standalone NOVAE Phase 0/1 zero-shot pilot. "
                     "reference=all uses the whole input and cannot feed confirmatory held-out classification.")
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--slide-key", required=True, help="Hard graph boundary (skin: sample_id).")
    parser.add_argument("--group-key", default=None, help="Optional grouping metadata, e.g. patient.")
    parser.add_argument("--technology", default="visium")
    parser.add_argument("--model-source", default="prism-oncology/novae-human-0")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--expression-mode", required=True, choices=("raw_counts", "preprocessed"),
                        help="Explicitly declare whether X contains raw counts or is already preprocessed.")
    parser.add_argument("--resolutions", type=float, nargs="+", default=None,
                        help="Pre-specified zero-shot Leiden resolutions (default: 0.5 1.0 2.0).")
    parser.add_argument("--resolution", type=float, default=None, help="Deprecated single-resolution alias for --resolutions.")
    parser.add_argument("--primary-resolution", type=float, default=None,
                        help="Predeclared primary resolution; it must be in --resolutions.")
    parser.add_argument("--expected-neighbor-distance-um", type=float, default=100.0,
                        help="Visium nominal center spacing in microns (default: 100).")
    parser.add_argument("--neighbor-distance-relative-tolerance", type=float, default=0.5,
                        help="Allowed relative deviation of each slide median distance (default: 0.5).")
    parser.add_argument("--graph-radius-um", type=float, default=None,
                        help="Optional positive physical radius for post-construction Visium graph pruning.")
    parser.add_argument("--min-domain-assignment-coverage", type=float,
                        default=DEFAULT_DOMAIN_ASSIGNMENT_COVERAGE,
                        help="Minimum NOVAE domain assignment coverage overall and per slide (default: 0.70).")
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coordinate-strategy", required=True,
                        choices=("materialized_microns", "shared_scalar", "visium_manifest"))
    parser.add_argument("--scale-to-microns", type=float, default=None)
    parser.add_argument("--sample-manifest", "--manifest", dest="sample_manifest", type=Path)
    parser.add_argument("--physical-spot-diameter-um", "--physical-spot-diameter", dest="physical_spot_diameter_um", type=float)
    parser.add_argument("--audit-only", action="store_true", help="Run Phase 0 audit without loading NOVAE/model.")
    return parser


def _run_impl(args: argparse.Namespace) -> int:
    if sys.version_info < (3, 11):
        raise NovaPilotError("NOVAE pilot requires Python >=3.11")
    validate_dataset_id(args.dataset_id)
    ad = _require_anndata()
    default_resolutions = [0.5, 1.0, 2.0]
    if args.resolution is not None and args.resolutions is not None:
        raise NovaPilotError("use --resolutions or deprecated --resolution, not both")
    if args.resolution is not None:
        requested = [args.resolution]
        primary_arg = args.primary_resolution if args.primary_resolution is not None else args.resolution
    else:
        requested = args.resolutions if args.resolutions is not None else default_resolutions
        primary_arg = args.primary_resolution if args.primary_resolution is not None else 1.0
    resolutions, primary_resolution = normalize_resolutions(requested, primary_arg)
    if args.workers < 0:
        raise NovaPilotError("workers must be non-negative")
    graph_radius_um = getattr(args, "graph_radius_um", None)
    if graph_radius_um is not None:
        try:
            graph_radius_um = float(graph_radius_um)
        except (TypeError, ValueError) as exc:
            raise NovaPilotError("graph radius must be numeric") from exc
        if not np.isfinite(graph_radius_um) or graph_radius_um <= 0:
            raise NovaPilotError("graph radius must be finite and positive")
    args.graph_radius_um = graph_radius_um
    try:
        min_coverage = float(args.min_domain_assignment_coverage)
    except (TypeError, ValueError) as exc:
        raise NovaPilotError("minimum domain-assignment coverage must be numeric") from exc
    if not np.isfinite(min_coverage) or min_coverage < 0 or min_coverage > 1:
        raise NovaPilotError("minimum domain-assignment coverage must be in [0, 1]")
    args.min_domain_assignment_coverage = min_coverage
    args.resolutions = resolutions
    args.primary_resolution = primary_resolution
    if args.coordinate_strategy == "shared_scalar":
        if args.scale_to_microns is None or not np.isfinite(args.scale_to_microns) or args.scale_to_microns <= 0:
            raise NovaPilotError("shared_scalar requires one positive --scale-to-microns")
        if args.sample_manifest or args.physical_spot_diameter_um is not None:
            raise NovaPilotError("shared_scalar cannot be combined with manifest conversion options")
    elif args.coordinate_strategy == "visium_manifest":
        if args.scale_to_microns is not None or args.sample_manifest is None or args.physical_spot_diameter_um is None:
            raise NovaPilotError("visium_manifest requires --sample-manifest and --physical-spot-diameter-um, and no scalar")
    elif args.scale_to_microns is not None or args.sample_manifest or args.physical_spot_diameter_um is not None:
        raise NovaPilotError("materialized_microns cannot be combined with calibration options")

    input_hash = sha256_file(args.input_h5ad)
    manifest_hash = sha256_file(args.sample_manifest) if args.sample_manifest else None
    source = ad.read_h5ad(args.input_h5ad)
    source_obs = source.obs_names.copy()
    source_vars = source.var_names.copy()
    snapshot = _nmf_snapshot(source)
    adata = source.copy()
    audit = validate_input(adata, args.slide_key, args.group_key)
    expression_before = expression_audit(adata, args.expression_mode)
    original_x = _copy_matrix(adata.X)
    counts_before_present = "counts" in adata.layers
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.coordinate_strategy == "visium_manifest":
        coordinate_audit = harmonize_visium_coordinates(
            adata, args.slide_key, args.sample_manifest, args.physical_spot_diameter_um
        )
        effective_scale = 1.0
    else:
        coordinate_audit = pd.DataFrame([{
            "slide": slide, "n_obs": count, "coordinate_strategy": args.coordinate_strategy,
            "microns_per_pixel": args.scale_to_microns if args.coordinate_strategy == "shared_scalar" else 1.0,
        } for slide, count in audit["slide_counts"].items()])
        effective_scale = args.scale_to_microns if args.coordinate_strategy == "shared_scalar" else 1.0
    audit["coordinate_strategy"] = args.coordinate_strategy
    audit["coordinate_representation"] = "materialized_microns" if args.coordinate_strategy != "shared_scalar" else "source_pixels_plus_novae_scalar"
    audit["coordinate_obsm_key"] = SPATIAL_KEY
    audit["coordinate_axis_order"] = ["x", "y"]
    audit["effective_novae_scale_to_microns"] = effective_scale
    audit["original_coordinate_obsm_key"] = ORIGINAL_SPATIAL_KEY if args.coordinate_strategy == "visium_manifest" else None
    audit["coordinate_audit_per_slide"] = coordinate_audit.to_dict(orient="records")
    audit["expression_mode"] = args.expression_mode
    audit["expression_audit"] = expression_before
    audit["novae_auto_preprocessing"] = args.expression_mode == "raw_counts"
    audit["manifest_sha256"] = manifest_hash
    audit["graph_radius_um_requested"] = args.graph_radius_um
    audit["graph_radius_pruning_applied"] = False
    atomic_json(args.output_dir / f"novae_coordinate_audit_{args.dataset_id}.json", audit)
    atomic_csv(coordinate_audit, args.output_dir / f"novae_coordinate_audit_{args.dataset_id}.csv")
    if list(adata.obs_names) != list(source_obs) or list(adata.var_names) != list(source_vars):
        raise NovaPilotError("coordinate adapter changed source observation or variable order")
    if args.audit_only:
        audit_runtime = _runtime_provenance()
        audit_provenance = {
            "analysis_scope": "exploratory", "phase": "0", "reference": "all",
            "audit_only": True, "confirmatory_held_out_classification_allowed": False,
            "input_sha256": input_hash, "dataset_id": args.dataset_id,
            "requested_resolutions": {resolution_token(value): value for value in resolutions},
            "primary_resolution": primary_resolution,
            "expected_neighbor_distance_um": args.expected_neighbor_distance_um,
            "neighbor_distance_relative_tolerance": args.neighbor_distance_relative_tolerance,
            "radius_pruning": {"requested_radius_um": args.graph_radius_um, "applied": False,
                               "reason": "audit_only"} if args.graph_radius_um is not None else None,
            "continued_exploratory_warning": "reference=all uses the whole cohort; labels are exploratory only",
            "slide_key": args.slide_key, "group_key": args.group_key,
            "slide_group_mapping": audit.get("slide_group_mapping"),
            "coordinate_strategy": args.coordinate_strategy,
            "coordinate_representation": audit["coordinate_representation"],
            "coordinate_input_unit": "microns" if args.coordinate_strategy == "materialized_microns" else "pixels",
            "coordinate_target_unit": "microns", "effective_novae_scale_to_microns": effective_scale,
            "coordinate_audit_per_slide": coordinate_audit.to_dict(orient="records"),
            "neighborhood_valid_key": NEIGHBORHOOD_VALID_KEY,
            "minimum_domain_assignment_coverage": args.min_domain_assignment_coverage,
            "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
            "sample_manifest_sha256": manifest_hash,
            "expression_mode": args.expression_mode,
            "expression_audit_before": expression_before,
            "expression_audit_after": expression_before,
            "novae_auto_preprocessing": args.expression_mode == "raw_counts",
            "auto_preprocessing_enabled": args.expression_mode == "raw_counts",
            "auto_preprocessing_applied": False, "counts_layer_preserved": None,
            "python": sys.version, "platform": platform.platform(),
            "runtime": audit_runtime,
            "runtime_packages": audit_runtime["packages"],
            "torch_runtime": audit_runtime.get("torch"),
            "resolved_settings": {key: str(value) for key, value in vars(args).items()},
        }
        audit_artifacts = {
            "coordinate_audit_json": args.output_dir / f"novae_coordinate_audit_{args.dataset_id}.json",
            "coordinate_audit_csv": args.output_dir / f"novae_coordinate_audit_{args.dataset_id}.csv",
        }
        audit_provenance = _canonicalize_provenance(audit_provenance)
        envelope = {"run": audit_provenance, "artifacts": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in audit_artifacts.items()
        }}
        atomic_json(args.output_dir / f"novae_provenance_{args.dataset_id}.json", envelope)
        atomic_json(args.output_dir / f"novae_resolved_manifest_{args.dataset_id}.json", envelope)
        return 0

    _seed_everything(args.seed)
    try:
        import novae
    except ImportError as exc:
        raise NovaPilotError("novae==1.1.1 is required for inference") from exc
    package_version = str(getattr(novae, "__version__", "unknown"))
    if package_version != "1.1.1":
        raise NovaPilotError(f"expected novae==1.1.1, found {package_version}")
    _configure_scale(novae, effective_scale)
    effective_auto_preprocessing = _configure_preprocessing(novae, args.expression_mode)
    model_path = getattr(args, "_resolved_model_path", None)
    model_provenance = getattr(args, "_resolved_model_provenance", None)
    if model_path is None or model_provenance is None:
        # Normal CLI execution resolves before staging; this fallback keeps the
        # implementation directly callable in focused tests.
        model_path, model_provenance = _resolve_model(args.model_source, args.model_revision)
    _add_checkpoint_hash(model_provenance)
    # NOVAE initializes its internal categorical slide IDs by default. Preserve
    # the user's column while validating the internal graph boundary.
    original_slide_values = adata.obs[args.slide_key].copy(deep=True)
    novae.spatial_neighbors(adata, slide_key=args.slide_key, technology=args.technology,
                            verbose=True)
    internal_slide_key = next(
        (key for key in ("novae_sid", "novae_slide_id") if key in adata.obs), args.slide_key
    )
    internal_graph_check = validate_spatial_graph(adata, internal_slide_key)
    _restore_obs_column(adata, args.slide_key, original_slide_values)
    if list(adata.obs[args.slide_key].astype(object)) != list(original_slide_values.astype(object)):
        raise NovaPilotError("NOVAE changed the user's original slide column")
    graph_check = validate_spatial_graph(adata, args.slide_key)
    graph_pre_pruning = graph_check
    radius_pruning = None
    if args.graph_radius_um is not None:
        radius_pruning = prune_spatial_graph_radius(
            adata, args.slide_key, args.graph_radius_um, effective_scale,
        )
        graph_check = validate_spatial_graph(adata, args.slide_key)
        audit["graph_radius_pruning_applied"] = True
        audit["graph_radius_pruning"] = radius_pruning
        atomic_json(args.output_dir / f"novae_coordinate_audit_{args.dataset_id}.json", audit)
    distance_qc = None
    if (args.technology.lower() == "visium" and args.expected_neighbor_distance_um is not None
            and args.coordinate_strategy != "shared_scalar"):
        distance_qc = neighbor_distance_calibration(
            adata, args.slide_key, args.expected_neighbor_distance_um,
            args.neighbor_distance_relative_tolerance,
        )
    model = novae.Novae.from_pretrained(str(model_path))
    hparams = getattr(model, "hparams", None)
    def _hparam(name: str) -> Any:
        if hparams is None:
            return None
        if isinstance(hparams, Mapping):
            return hparams.get(name)
        return getattr(hparams, name, None)
    checkpoint_hparams = {"n_hops_local": _hparam("n_hops_local"), "n_hops_view": _hparam("n_hops_view")}
    if args.technology.lower() == "visium":
        for name, value in checkpoint_hparams.items():
            if value is not None and int(value) not in {1, 2}:
                raise NovaPilotError(f"Visium checkpoint {name} must be 1 or 2, found {value}")
    model.compute_representations(adata, zero_shot=True, reference="all",
                                  accelerator=args.accelerator, num_workers=args.workers)
    latent_key = next((key for key in adata.obsm.keys() if key == "novae_latent" or key.startswith("novae_latent")), None)
    if latent_key is None:
        raise NovaPilotError("NOVAE did not produce a latent representation")
    _neighborhood_valid_mask(adata)
    latent_dim = int(np.asarray(adata.obsm[latent_key]).shape[1])
    domain_keys: dict[float, str] = {}
    returned_domain_keys: dict[float, str] = {}
    domain_assignment_audits: dict[float, dict[str, Any]] = {}
    metrics_rows: list[dict[str, Any]] = []
    for resolution in resolutions:
        previous_values = {key: adata.obs[key].copy() for key in domain_keys.values() if key in adata.obs}
        returned_key = model.assign_domains(adata, resolution=resolution)
        if not isinstance(returned_key, str) or returned_key not in adata.obs:
            raise NovaPilotError(f"NOVAE did not produce expected domain key for resolution {resolution}")
        domain_key = returned_key
        # Preserve a prior assignment if a wrapper reuses its key. Normal NOVAE
        # v1.1.1 keys are already resolution-specific and remain untouched.
        if returned_key in previous_values:
            current_values = adata.obs[returned_key].copy()
            adata.obs[returned_key] = previous_values[returned_key]
            domain_key = f"novae_domains_res{resolution}"
            adata.obs[domain_key] = current_values
        _validate_domain_column(adata, domain_key)
        assignment_audit = audit_domain_assignments(
            adata, args.slide_key, domain_key, args.min_domain_assignment_coverage,
        )
        domain_keys[resolution] = domain_key
        returned_domain_keys[resolution] = returned_key
        domain_assignment_audits[resolution] = assignment_audit
        metric_values = _science_metrics(novae, adata, args.slide_key, domain_key)
        metrics_rows.append({"resolution": resolution, "primary": bool(np.isclose(resolution, primary_resolution, rtol=0, atol=1e-12)),
                             "domain_key": domain_key, **metric_values,
                             "interpretation": "comparative; not an absolute pass/fail metric",
                             "fide_interpretation": "high FIDE indicates spatial continuity",
                             "jsd_interpretation": "JSD across slides should not be minimized blindly; composition may differ biologically"})
    latent_dim = _validate_inference_outputs(adata, domain_keys[resolutions[0]])
    _restore_obs_column(adata, args.slide_key, original_slide_values)
    if list(adata.obs[args.slide_key].astype(object)) != list(original_slide_values.astype(object)):
        raise NovaPilotError("NOVAE changed the user's original slide column after inference")
    graph_check = validate_spatial_graph(adata, args.slide_key)
    diagnostics = graph_diagnostics(adata, args.slide_key)
    expression_after = post_inference_expression_audit(
        adata, args.expression_mode, original_x, effective_auto_preprocessing, counts_before_present
    )
    latent_key = next(key for key in adata.obsm.keys() if key == "novae_latent" or key.startswith("novae_latent"))
    # Latents and graph diagnostics are representation-level outputs and are
    # intentionally computed once, not once per resolution.
    latent = latent_summary(adata, args.slide_key, latent_key)
    group_latent = latent_summary(adata, args.group_key, latent_key) if args.group_key else None
    domain_summary = _domain_resolution_summary(
        adata, args.slide_key, resolutions, domain_keys, primary_resolution,
        domain_assignment_audits,
    )
    if list(adata.obs_names) != list(source_obs) or list(adata.var_names) != list(source_vars):
        raise NovaPilotError("NOVAE changed source observation or variable order")
    assert_nmf_unchanged(adata, snapshot)
    model_state_dir = args.output_dir / "novae_zero_shot_model"
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        try:
            save_pretrained(model_state_dir)
            model_state = {"supported": True, "path": model_state_dir.name,
                           "files_sha256": sha256_directory(model_state_dir),
                           "note": "contains cohort-derived zero-shot prototype weights; exploratory only; may support cheap assign_domains on this annotated cohort without re-encoding, but must not initialize held-out confirmatory analysis"}
            model_state["sha256"] = hashlib.sha256(json.dumps(model_state["files_sha256"], sort_keys=True).encode()).hexdigest()
        except Exception as exc:
            raise NovaPilotError(f"NOVAE save_pretrained failed: {exc}") from exc
    else:
        model_state = {"supported": False, "path": None,
                       "note": "model API did not expose save_pretrained; all requested resolutions are already emitted"}
    runtime_provenance = _runtime_provenance(package_version)
    science_metrics_provenance = {
        resolution_token(row["resolution"]): {key: value for key, value in row.items() if key != "resolution"}
        for row in metrics_rows
    }
    distance_qc_provenance = (
        {str(row["slide"]): {key: value for key, value in row.items() if key != "slide"}
         for row in distance_qc.to_dict(orient="records")}
        if distance_qc is not None else None
    )
    core_run_provenance: dict[str, Any] = {
        "analysis_scope": "exploratory", "reference": "all", "inference_mode": "zero_shot",
        "continued_exploratory_warning": "reference=all uses the whole cohort; labels are exploratory only",
        "confirmatory_held_out_classification_allowed": False,
        "audit_only": False,
        "input_h5ad": str(args.input_h5ad), "input_sha256": input_hash,
        "dataset_id": args.dataset_id, "slide_key": args.slide_key, "group_key": args.group_key,
        "technology": args.technology, "coordinate_strategy": args.coordinate_strategy,
        "expression_mode": args.expression_mode,
        "expression_audit_before": expression_before,
        "expression_audit_after": expression_after,
        "novae_auto_preprocessing": effective_auto_preprocessing,
        "auto_preprocessing_enabled": effective_auto_preprocessing,
        "auto_preprocessing_applied": expression_after["auto_preprocessing_applied"],
        "counts_layer_present": expression_after["counts_layer_present"],
        "counts_layer_created": expression_after["counts_layer_created"],
        "counts_layer_preserved": expression_after["counts_layer_preserved"],
        "effective_novae_scale_to_microns": effective_scale,
        "requested_scale_to_microns": args.scale_to_microns,
        "physical_spot_diameter_um": args.physical_spot_diameter_um,
        "duplicate_coordinate_policy": "reject_within_slide",
        "coordinate_input_unit": "microns" if args.coordinate_strategy == "materialized_microns" else "pixels",
        "coordinate_target_unit": "microns",
        "coordinate_obsm_key": SPATIAL_KEY,
        "coordinate_axis_order": "x,y",
        "sample_manifest": str(args.sample_manifest) if args.sample_manifest else None,
        "sample_manifest_sha256": manifest_hash,
        "model": model_provenance,
        "model_source": args.model_source,
        "model_revision": args.model_revision,
        "resolved_model_revision": model_provenance.get("resolved_revision"),
        "checkpoint_sha256": model_provenance.get("checkpoint_sha256"),
        "package_version": package_version,
        "python": sys.version, "platform": platform.platform(),
        "runtime": runtime_provenance,
        "runtime_packages": runtime_provenance["packages"],
        "torch_runtime": runtime_provenance.get("torch"),
        "accelerator": args.accelerator, "device": _runtime_device(args.accelerator), "workers": args.workers, "seed": args.seed,
        "requested_resolutions": {resolution_token(value): value for value in resolutions},
        "primary_resolution": primary_resolution,
        "neighborhood_valid_key": NEIGHBORHOOD_VALID_KEY,
        "minimum_domain_assignment_coverage": args.min_domain_assignment_coverage,
        "domain_assignment_audits": {
            resolution_token(resolution): audit for resolution, audit in domain_assignment_audits.items()
        },
        "primary_resolution_is_predeclared": True, "domain_keys": {str(k): v for k, v in domain_keys.items()},
        "returned_domain_keys": {str(k): v for k, v in returned_domain_keys.items()},
        "representation_computation_count": 1, "latent_key": latent_key, "latent_dimension": latent_dim,
        "checkpoint_hparams": checkpoint_hparams, "zero_shot_model_artifact": model_state,
        "zero_shot_model_scientific_note": "compute_representations(zero_shot=True, reference=all) updates prototypes from the cohort; saved state is contaminated exploratory state and is not a confirmatory held-out initializer",
        "science_metrics": science_metrics_provenance, "distance_qc": distance_qc_provenance,
        "slide_group_mapping": audit.get("slide_group_mapping"),
        "distance_qc_expected_um": args.expected_neighbor_distance_um,
        "distance_qc_relative_tolerance": args.neighbor_distance_relative_tolerance,
        "domain_key": domain_keys[primary_resolution], "graph": graph_check,
        "graph_pre_pruning": graph_pre_pruning,
        "internal_slide_key": internal_slide_key,
        "internal_graph": internal_graph_check,
        "graph_topology": "per_slide; no_cross_slide_edges",
        "radius_pruning": radius_pruning,
        "resolved_settings": {key: str(value) for key, value in vars(args).items()},
        "source_row_order_preserved": list(adata.obs_names) == list(source_obs),
        "nmf_columns_preserved": True,
    }
    # The H5AD embeds the immutable core only. Canonicalize once before both
    # representations because AnnData versions may omit None mapping values.
    # Final artifact hashes belong in the external envelope, avoiding the
    # impossible self-hash/circular update.
    core_run_provenance = _canonicalize_core_provenance(core_run_provenance)
    adata.uns["novae_pilot_provenance"] = core_run_provenance
    output_h5ad = args.output_dir / f"novae_{args.dataset_id}_zero_shot.h5ad"
    atomic_h5ad(adata, output_h5ad)
    artifact_paths = {
        "annotated_h5ad": output_h5ad,
        "coordinate_audit_json": args.output_dir / f"novae_coordinate_audit_{args.dataset_id}.json",
        "coordinate_audit_csv": args.output_dir / f"novae_coordinate_audit_{args.dataset_id}.csv",
        "graph_diagnostics_csv": args.output_dir / f"novae_graph_diagnostics_{args.dataset_id}.csv",
        "latent_summary_csv": args.output_dir / f"novae_latent_summary_{args.dataset_id}.csv",
        "domain_resolution_summary_csv": args.output_dir / f"novae_domain_resolution_summary_{args.dataset_id}.csv",
        "science_metrics_csv": args.output_dir / f"novae_science_metrics_{args.dataset_id}.csv",
    }
    atomic_csv(diagnostics, artifact_paths["graph_diagnostics_csv"])
    atomic_csv(latent, artifact_paths["latent_summary_csv"])
    atomic_csv(domain_summary, artifact_paths["domain_resolution_summary_csv"])
    atomic_csv(pd.DataFrame(metrics_rows), artifact_paths["science_metrics_csv"])
    if distance_qc is not None:
        distance_path = args.output_dir / f"novae_neighbor_distance_qc_{args.dataset_id}.csv"
        atomic_csv(distance_qc, distance_path)
        artifact_paths["neighbor_distance_qc_csv"] = distance_path
    for resolution, domain_key in domain_keys.items():
        token = resolution_token(resolution)
        adjacency_path = args.output_dir / f"novae_domain_adjacency_{args.dataset_id}_res-{token}.csv"
        proportions_path = args.output_dir / f"novae_domain_proportions_{args.dataset_id}_res-{token}.csv"
        atomic_csv(domain_adjacency(adata, args.slide_key, domain_key), adjacency_path)
        atomic_csv(domain_proportions(adata, args.slide_key, domain_key), proportions_path)
        artifact_paths[f"domain_adjacency_res_{token}_csv"] = adjacency_path
        artifact_paths[f"domain_proportions_res_{token}_csv"] = proportions_path
        if args.group_key:
            group_path = args.output_dir / f"novae_group_domain_proportions_{args.dataset_id}_res-{token}.csv"
            atomic_csv(domain_proportions(adata, args.group_key, domain_key), group_path)
            artifact_paths[f"group_domain_proportions_res_{token}_csv"] = group_path
    if group_latent is not None:
        latent_group_path = args.output_dir / f"novae_group_latent_summary_{args.dataset_id}.csv"
        atomic_csv(group_latent, latent_group_path)
        artifact_paths["group_latent_summary_csv"] = latent_group_path
    if model_state.get("supported"):
        artifact_paths["zero_shot_model_dir"] = model_state_dir
    artifact_hashes = {
        name: {"path": str(path.name), "sha256": _artifact_sha256(path)}
        for name, path in artifact_paths.items()
    }
    envelope = {"run": _canonicalize_provenance(core_run_provenance), "artifacts": artifact_hashes}
    atomic_json(args.output_dir / f"novae_provenance_{args.dataset_id}.json", envelope)
    atomic_json(args.output_dir / f"novae_resolved_manifest_{args.dataset_id}.json", envelope)
    return 0


def _run(args: argparse.Namespace) -> int:
    """Run into a unique staging sibling and publish only on complete success."""
    final_dir = args.output_dir
    if final_dir.exists():
        raise NovaPilotError(f"output directory already exists; use a new path or remove it: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_path = resolved_model_provenance = None
    if not args.audit_only:
        resolved_model_path, resolved_model_provenance = _resolve_model(args.model_source, args.model_revision)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.staging-", dir=final_dir.parent))
    staged_args = argparse.Namespace(**vars(args))
    staged_args.output_dir = staging_dir
    staged_args._resolved_model_path = resolved_model_path
    staged_args._resolved_model_provenance = resolved_model_provenance
    try:
        result = _run_impl(staged_args)
        if final_dir.exists():
            raise NovaPilotError(f"output directory appeared during run: {final_dir}")
        staging_dir.rename(final_dir)
        return result
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# Small public aliases keep the audit helpers convenient for notebooks/tests
# without adding another implementation layer.
validate_graph = validate_spatial_graph
validate_resolutions = normalize_resolutions
compute_neighbor_distance_qc = neighbor_distance_calibration
prune_graph_radius = prune_spatial_graph_radius
compute_domain_adjacency = domain_adjacency
compute_domain_proportions = domain_proportions
audit_novae_domains = audit_domain_assignments
domain_assignment_audit = audit_domain_assignments
validate_domain_assignments = audit_domain_assignments
summarize_latents = latent_summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except (NovaPilotError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
