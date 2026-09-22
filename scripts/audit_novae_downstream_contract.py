#!/usr/bin/env python3
"""Read-only audit of the skin pseudo-FOV/NMF downstream contract.

The audit is deliberately boring: it inventories artifacts, validates row and
index contracts, and never selects a candidate using a model metric. Real H5AD
reads belong in the generated CPU SLURM job; tests use synthetic H5ADs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

HISTORICAL_SOURCE = "/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_1mmfov_spatial.h5ad"
HISTORICAL_RUN = "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs"
FULLSWEEP_SOURCE = "/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_1000umfov_spatial.h5ad"
FULLSWEEP_RUN = "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1000umfov_poisson75_fullsweep/outputs"


@dataclass(frozen=True)
class Candidate:
    name: str
    source_h5ad: Path
    run_output: Path


DEFAULT_CANDIDATES = (
    Candidate("historical_164", Path(HISTORICAL_SOURCE), Path(HISTORICAL_RUN)),
    Candidate("fullsweep_225", Path(FULLSWEEP_SOURCE), Path(FULLSWEEP_RUN)),
)

# The first matching path is recorded. Provenance is intentionally optional:
# old runs predate a uniform manifest convention, but its absence is visible.
ARTIFACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "nmf_h5ad": ("cosmx_with_nmf.h5ad",),
    "post_nmf_obs": ("post_nmf_obs.csv",),
    "enrichment": (
        "enrichment_features_fov.csv", "enrichment_features_fov.parquet",
        "enrichment_fov.csv", "enrichment_fov.parquet",
    ),
    "niche_gene": (
        "niche_gene_features_fov.csv", "niche_gene_features_fov.parquet",
        "niche_gene_fov.csv", "niche_gene_fov.parquet",
    ),
    "combined": (
        "MLP_FOVFeatures_inputs/combined_features_filtered.parquet",
        "combined_features_filtered.parquet",
    ),
    "targets": (
        "MLP_FOVFeatures_inputs/targets_y.parquet", "targets_y.parquet",
    ),
    "groups": (
        "MLP_FOVFeatures_inputs/groups.parquet", "groups.parquet",
    ),
}
REQUIRED_ARTIFACTS = tuple(ARTIFACT_PATTERNS)


class ContractAuditError(ValueError):
    """A malformed candidate or contract violation."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str, *, required: bool = True) -> bool:
    checks.append({"check": name, "passed": bool(passed), "required": required, "detail": detail})
    return passed


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _clean(values: Iterable[Any], label: str) -> list[str]:
    result = []
    for value in values:
        if _missing(value) or not str(value).strip() or str(value).strip().lower() == "nan":
            raise ContractAuditError(f"{label} contains missing/blank values")
        result.append(str(value).strip())
    return result


def _unique(values: list[str], label: str) -> tuple[bool, str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    duplicate_sample = sorted(duplicates)
    return (not duplicate_sample, f"{label}: {len(values)} values; duplicate_sample={duplicate_sample[:10]}")


def _pick(columns: Iterable[str], candidates: tuple[str, ...], label: str) -> str:
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ContractAuditError(f"{label} missing; accepted keys={list(candidates)}")


def _disease_key(columns: Iterable[str]) -> str:
    # Disease_State is canonical. Lowercase is an explicit compatibility path,
    # not a case-insensitive mutation of the source table.
    return _pick(
        columns,
        ("Disease_State", "disease_state", "Disease/Health State", "Disease.Health.State"),
        "Disease_State",
    )


def _fov_values(frame: pd.DataFrame) -> tuple[list[str], str]:
    patient_key = _pick(frame.columns, ("patient", "Patient", "subject", "sample_id"), "patient")
    patients = _clean(frame[patient_key], patient_key)
    for key in ("field_of_view", "fov_key", "unique_fov", "field_of_view_id"):
        if key in frame.columns:
            return _clean(frame[key], key), key
    for key in ("fov", "FOV"):
        if key in frame.columns:
            raw = _clean(frame[key], key)
            return [f"{patient}_{fov}" for patient, fov in zip(patients, raw, strict=True)], f"{patient_key}+{key}"
    raise ContractAuditError("missing field_of_view or patient+fov mapping keys")


def _obs_ids(adata: Any, label: str) -> list[str]:
    ids = _clean(list(adata.obs_names), f"{label}.obs_names")
    ok, detail = _unique(ids, f"{label}.obs_names")
    if not ok:
        raise ContractAuditError(detail)
    return ids


def _validate_adata_obs(adata: Any, label: str, *, nmf: bool = False) -> dict[str, Any]:
    frame = adata.obs.copy()
    patient_key = _pick(frame.columns, ("patient", "Patient", "subject", "sample_id"), f"{label} patient")
    disease_key = _disease_key(frame.columns)
    unique_key = _pick(
        frame.columns,
        ("unique_cell_id", "unique_cellid", "cell_id", "cell_ID"),
        f"{label} unique_cell_id",
    )
    fov_values, fov_mapping = _fov_values(frame)
    patient = _clean(frame[patient_key], f"{label}.{patient_key}")
    disease = _clean(frame[disease_key], f"{label}.{disease_key}")
    unique = _clean(frame[unique_key], f"{label}.{unique_key}")
    ok, detail = _unique(unique, f"{label}.{unique_key}")
    if not ok:
        raise ContractAuditError(detail)
    nmf_factor: list[str] | None = None
    if nmf:
        for key in ("NMF_factor", "dominant_nmf_factor"):
            if key not in frame.columns:
                raise ContractAuditError(f"NMF output missing obs[{key!r}]")
            values = _clean(frame[key], f"{label}.{key}")
            if key == "NMF_factor":
                nmf_factor = values
    return {
        "frame": frame,
        "patient_key": patient_key,
        "disease_key": disease_key,
        "unique_cell_key": unique_key,
        "fov_mapping": fov_mapping,
        "patient": patient,
        "disease": disease,
        "unique_cell_id": unique,
        "fov": fov_values,
        "nmf_factor": nmf_factor,
    }


def _discover_artifacts(run_output: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    inventory: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for name, patterns in ARTIFACT_PATTERNS.items():
        selected = next((run_output / pattern for pattern in patterns if (run_output / pattern).is_file()), None)
        item = {
            "artifact": name,
            "required": True,
            "present": selected is not None,
            "path": str(selected) if selected else "",
            "accepted_names": ";".join(patterns),
        }
        inventory[name] = item
        rows.append(item)
    provenance = sorted(
        path for path in run_output.rglob("*")
        if path.is_file() and re.search(r"(manifest|provenance|artifact)", path.name, re.I)
    ) if run_output.is_dir() else []
    inventory["provenance_manifest"] = {
        "artifact": "provenance_manifest", "required": False, "present": bool(provenance),
        "path": str(provenance[0]) if provenance else "",
        "paths": ";".join(str(path) for path in provenance),
        "accepted_names": "*manifest*/*provenance*/*artifact*",
    }
    rows.append(inventory["provenance_manifest"])
    return inventory, rows


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        for key in ("field_of_view", "fov_key", "fov", "unique_fov"):
            if key in raw.columns:
                return raw.set_index(key)
        if raw.columns[0].lower().startswith("unnamed"):
            return raw.set_index(raw.columns[0]).rename_axis(None)
        raise ContractAuditError(f"{path.name} has no field_of_view/index column")
    try:
        return pd.read_parquet(path)
    except (ImportError, OSError, ValueError) as exc:
        raise ContractAuditError(f"could not read parquet artifact {path}: {exc}") from exc


def _table_index(frame: pd.DataFrame, label: str) -> list[str]:
    if frame.index.nlevels != 1:
        raise ContractAuditError(f"{label} index must be one-dimensional")
    values = _clean(frame.index.tolist(), f"{label} index")
    ok, detail = _unique(values, f"{label} index")
    if not ok:
        raise ContractAuditError(detail)
    return values


def _index_comparison(expected: list[str], observed: list[str], label: str) -> dict[str, Any]:
    expected_set, observed_set = set(expected), set(observed)
    return {
        "artifact": label,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "intersection_count": len(expected_set & observed_set),
        "missing": sorted(expected_set - observed_set),
        "extra": sorted(observed_set - expected_set),
        "order_equal": expected == observed,
        "set_equal": expected_set == observed_set,
        "exact": expected == observed,
        "contains_expected": expected_set <= observed_set,
    }


def _table_values(frame: pd.DataFrame, label: str) -> list[str]:
    if frame.shape[1] == 0:
        raise ContractAuditError(f"{label} has no value columns")
    return _clean(frame.iloc[:, 0], f"{label} values")


_METADATA_COLUMNS = frozenset({
    "patient", "Patient", "subject", "sample_id", "Disease_State", "disease_state",
    "Disease/Health State", "Disease.Health.State", "field_of_view", "fov_key", "fov",
    "FOV", "unique_fov", "field_of_view_id",
})


def _feature_frame(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ContractAuditError(f"{label} has duplicate columns: {duplicate_columns[:10]}")
    metadata = [str(column) for column in frame.columns if str(column) in _METADATA_COLUMNS]
    features = frame.drop(columns=metadata, errors="ignore")
    if features.shape[1] == 0:
        raise ContractAuditError(f"{label} has no feature columns after metadata removal={metadata}")
    return features, metadata


def _finite_features(frame: pd.DataFrame, label: str) -> tuple[bool, str]:
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        return False, f"{label} contains non-numeric feature values: {exc}"
    if not np.isfinite(values).all():
        return False, f"{label} contains non-finite feature values"
    return True, f"{label}: {frame.shape[1]} finite numeric features"


def _feature_checks(checks: list[dict[str, Any]], frame: pd.DataFrame, label: str) -> list[str]:
    try:
        features, metadata = _feature_frame(frame, label)
    except ContractAuditError as exc:
        detail = str(exc)
        if "duplicate columns" in detail:
            _check(checks, f"{label}_no_duplicate_columns", False, detail)
        _check(checks, f"{label}_columns", False, detail)
        return []
    ok, detail = _finite_features(features, label)
    _check(checks, f"{label}_finite_numeric", ok, detail)
    no_missing = not bool(features.isna().any().any())
    _check(checks, f"{label}_no_missing", no_missing, f"metadata_ignored={metadata}")
    _check(checks, f"{label}_no_duplicate_columns", True, "feature column labels are unique")
    return [str(column) for column in features.columns]


def _composition_table(post_fov: list[str], factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(pd.Series(post_fov, name="field_of_view"), pd.Series(factors, name="NMF_factor"))
    counts = counts.sort_index().sort_index(axis=1)
    proportions = counts.div(counts.sum(axis=1), axis=0).astype(float)
    proportions.columns = [f"nmf_prop_{column}" for column in proportions.columns]
    return counts, proportions


def _reconcile_composition(checks: list[dict[str, Any]], combined: pd.DataFrame, expected: pd.DataFrame) -> None:
    expected_columns = [str(column) for column in expected.columns]
    actual_columns = [str(column) for column in combined.columns]
    missing = [column for column in expected_columns if column not in actual_columns]
    unexpected = [column for column in actual_columns if column.startswith("nmf_prop_") and column not in expected_columns]
    _check(checks, "combined_composition_columns", not missing, f"expected={expected_columns} missing={missing}")
    _check(checks, "combined_no_unexpected_composition_columns", not unexpected, f"unexpected={unexpected}")
    if missing:
        return
    values = combined.loc[:, expected_columns].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=float)
    finite = bool(np.isfinite(array).all())
    nonnegative = bool((array >= 0).all()) if finite else False
    _check(checks, "combined_composition_finite_numeric", finite, "computed composition columns are finite numeric")
    _check(checks, "combined_composition_nonnegative", nonnegative, "computed composition columns are nonnegative")
    row_sums = values.sum(axis=1).to_numpy(dtype=float)
    positive = bool(np.isfinite(row_sums).all() and (row_sums > 0).all())
    normalized = bool(np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8)) if positive else False
    _check(checks, "combined_composition_positive_row_sums", positive, "composition row sums are positive")
    _check(checks, "combined_composition_normalized", normalized, "composition row sums are approximately one")
    expected_values = expected.reindex(combined.index)[expected_columns].to_numpy(dtype=float)
    reconciled = bool(np.allclose(array, expected_values, rtol=1e-6, atol=1e-8))
    _check(checks, "combined_composition_reconciles", reconciled, "combined NMF proportions match post_nmf_obs crosstab proportions")


def _close(adata: Any) -> None:
    close = getattr(adata, "close", None)
    if callable(close):
        close()


def audit_candidate(candidate: Candidate) -> dict[str, Any]:
    """Audit one candidate without writing to its source or run directory."""
    checks: list[dict[str, Any]] = []
    inventory, _ = _discover_artifacts(candidate.run_output)
    for item in inventory.values():
        if item["required"]:
            _check(
                checks, f"artifact:{item['artifact']}", bool(item["present"]),
                item["path"] or f"missing accepted names: {item['accepted_names']}",
            )
        else:
            _check(
                checks, "artifact:provenance_manifest", True,
                item["path"] if item["present"] else "not present (optional legacy artifact)",
                required=False,
            )

    source_info: dict[str, Any] = {}
    nmf_info: dict[str, Any] = {}
    resolved_keys: dict[str, Any] = {}
    row_intersections: dict[str, Any] = {}
    errors: list[str] = []
    source = nmf = None

    if not candidate.source_h5ad.is_file():
        _check(checks, "source_h5ad", False, f"missing: {candidate.source_h5ad}")
    else:
        _check(checks, "source_h5ad", True, str(candidate.source_h5ad))
    if not inventory["nmf_h5ad"]["present"]:
        _check(checks, "nmf_h5ad_readable", False, "cosmx_with_nmf.h5ad is missing")
    elif candidate.source_h5ad.is_file():
        try:
            try:
                import anndata as ad
            except ImportError as exc:
                raise RuntimeError("anndata is required to audit H5AD artifacts") from exc
            # backed='r' is intentional: no write-capable or in-memory mutation path.
            source = ad.read_h5ad(candidate.source_h5ad, backed="r")
            nmf = ad.read_h5ad(Path(inventory["nmf_h5ad"]["path"]), backed="r")
            source_ids = _obs_ids(source, "source")
            nmf_ids = _obs_ids(nmf, "NMF")
            source_info = _validate_adata_obs(source, "source")
            nmf_info = _validate_adata_obs(nmf, "NMF", nmf=True)
            resolved_keys["source"] = {key: source_info[key] for key in ("patient_key", "disease_key", "unique_cell_key", "fov_mapping")}
            resolved_keys["nmf"] = {key: nmf_info[key] for key in ("patient_key", "disease_key", "unique_cell_key", "fov_mapping")}
            _check(checks, "source_nmf_nonempty", bool(source_ids) and bool(nmf_ids), f"source={len(source_ids)} NMF={len(nmf_ids)}")
            source_set, nmf_set = set(source_ids), set(nmf_ids)
            row_intersections["obs_names"] = {
                "intersection": [value for value in source_ids if value in nmf_set],
                "source_only": sorted(source_set - nmf_set),
                "nmf_only": sorted(nmf_set - source_set),
            }
            _check(checks, "source_nmf_obs_order", source_ids == nmf_ids, f"source={len(source_ids)} NMF={len(nmf_ids)}")
            _check(checks, "source_nmf_obs_set", source_set == nmf_set, f"intersection={len(source_set & nmf_set)}")
            _check(checks, "source_obs_matches_unique_cell_id", source_ids == source_info["unique_cell_id"], "source obs_names preserve unique_cell_id")
            _check(checks, "nmf_obs_matches_unique_cell_id", nmf_ids == nmf_info["unique_cell_id"], "NMF obs_names preserve unique_cell_id")
            _check(checks, "source_nmf_unique_cell_order", source_info["unique_cell_id"] == nmf_info["unique_cell_id"], "unique_cell_id order is preserved")
            _check(checks, "source_nmf_patient_values", source_info["patient"] == nmf_info["patient"], "patient values are equal in observation order")
            _check(checks, "source_nmf_disease_state_values", source_info["disease"] == nmf_info["disease"], "disease-state values are equal in observation order")
            _check(checks, "source_nmf_fov_order", source_info["fov"] == nmf_info["fov"], f"source={source_info['fov_mapping']} NMF={nmf_info['fov_mapping']}")
        except (ContractAuditError, OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
            _check(checks, "h5ad_contract", False, str(exc))
        finally:
            _close(source)
            _close(nmf)

    post_fov: list[str] = []
    post_patient: list[str] = []
    post_disease: list[str] = []
    post_cell: list[str] = []
    post_factor: list[str] = []
    fov_rows: list[dict[str, Any]] = []
    composition_counts = pd.DataFrame()
    composition = pd.DataFrame()
    post_path = Path(inventory["post_nmf_obs"]["path"]) if inventory["post_nmf_obs"]["present"] else None
    if post_path is not None:
        try:
            post = pd.read_csv(post_path)
            post_fov, post_mapping = _fov_values(post)
            _check(checks, "post_nmf_obs_nonempty", bool(post_fov), f"cell rows={len(post_fov)}")
            post_patient_key = _pick(post.columns, ("patient", "Patient", "subject", "sample_id"), "post_nmf_obs patient")
            post_disease_key = _disease_key(post.columns)
            post_cell_key = _pick(post.columns, ("unique_cell_id", "unique_cellid", "cell_id", "cell_ID"), "post_nmf_obs unique_cell_id")
            if "NMF_factor" not in post.columns:
                raise ContractAuditError("post_nmf_obs missing NMF_factor")
            post_patient = _clean(post[post_patient_key], "post_nmf_obs patient")
            post_disease = _clean(post[post_disease_key], "post_nmf_obs disease")
            post_cell = _clean(post[post_cell_key], "post_nmf_obs unique_cell_id")
            post_factor = _clean(post["NMF_factor"], "post_nmf_obs NMF_factor")
            resolved_keys["post_nmf_obs"] = {
                "patient_key": post_patient_key, "disease_key": post_disease_key,
                "unique_cell_key": post_cell_key, "fov_mapping": post_mapping, "nmf_factor_key": "NMF_factor",
            }
            ids_unique, ids_detail = _unique(post_cell, "post_nmf_obs unique_cell_id")
            _check(checks, "post_nmf_obs_unique_cell_id", ids_unique, ids_detail)
            _check(checks, "post_nmf_obs_nmf_factor", True, "NMF_factor is present, nonmissing, and used for composition")

            if nmf_info.get("unique_cell_id") and nmf_info.get("nmf_factor") is not None:
                nmf_by_id = dict(zip(nmf_info["unique_cell_id"], nmf_info["nmf_factor"], strict=True))
                post_set, nmf_set = set(post_cell), set(nmf_by_id)
                _check(checks, "post_nmf_ids_nmf_set", post_set == nmf_set, f"intersection={len(post_set & nmf_set)} post_only={len(post_set - nmf_set)} nmf_only={len(nmf_set - post_set)}")
                factor_match = post_set == nmf_set and all(post_factor[index] == nmf_by_id.get(cell_id) for index, cell_id in enumerate(post_cell))
                _check(checks, "post_nmf_factors_match_nmf_by_id", factor_match, "post_nmf_obs NMF_factor matches NMF H5AD by unique_cell_id")
            else:
                _check(checks, "post_nmf_ids_nmf_set", False, "NMF H5AD identity/factor map unavailable")
                _check(checks, "post_nmf_factors_match_nmf_by_id", False, "NMF H5AD identity/factor map unavailable")

            composition_counts, composition = _composition_table(post_fov, post_factor)
            post_group = pd.DataFrame({"fov": post_fov, "patient": post_patient, "label": post_disease})
            fov_frame = post_group.groupby("fov", sort=False).agg(
                patient=("patient", lambda values: sorted(set(values))),
                label=("label", lambda values: sorted(set(values))),
                cell_count=("fov", "size"),
            )
            for fov in composition.index:
                patients = fov_frame.loc[fov, "patient"]
                labels = fov_frame.loc[fov, "label"]
                fov_rows.append({
                    "field_of_view": str(fov),
                    "patient": patients[0] if len(patients) == 1 else "",
                    "label": labels[0] if len(labels) == 1 else "",
                    "cell_count": int(fov_frame.loc[fov, "cell_count"]),
                    "one_patient": len(patients) == 1,
                    "one_label": len(labels) == 1,
                })
            fov_valid = all(row["one_patient"] and row["one_label"] for row in fov_rows)
            _check(checks, "fov_one_patient_one_label", fov_valid, f"FOVs={len(fov_rows)}")
            _check(checks, "fov_mapping_reconstructed", True, f"mapping={post_mapping}; composition FOV rows={len(composition)}")
            _check(checks, "composition_source_sorted_index", list(composition.index) == sorted(composition.index), "composition index follows crosstab().sort_index()")
        except (ContractAuditError, OSError, ValueError) as exc:
            errors.append(str(exc))
            _check(checks, "post_nmf_obs_contract", False, str(exc))
            composition_counts = pd.DataFrame()
            composition = pd.DataFrame()
    else:
        _check(checks, "post_nmf_obs_contract", False, "post_nmf_obs.csv is missing")

    table_frames: dict[str, pd.DataFrame] = {}
    table_indices: dict[str, list[str]] = {}
    feature_columns: dict[str, list[str]] = {}
    for name in ("enrichment", "niche_gene", "combined", "targets", "groups"):
        if not inventory[name]["present"]:
            continue
        try:
            table = _load_table(Path(inventory[name]["path"]))
            observed = _table_index(table, name)
            table.index = observed
            table_frames[name] = table
            table_indices[name] = observed
            _check(checks, f"{name}_no_duplicate_index", True, "index is unique")
            if name in ("enrichment", "niche_gene", "combined"):
                feature_columns[name] = _feature_checks(checks, table, name)
        except (ContractAuditError, OSError, ValueError) as exc:
            errors.append(str(exc))
            _check(checks, f"{name}_readable", False, str(exc))

    canonical: list[str] = table_indices.get("combined", [])
    composition_index = [str(value) for value in composition.index]
    fov_lookup = pd.DataFrame(fov_rows).set_index("field_of_view") if fov_rows else pd.DataFrame()
    structurally_zero_enrichment_ids: list[str] = []
    structurally_zero_enrichment_fatal_ids: list[str] = []
    if "combined" in table_frames:
        _check(checks, "combined_nonempty", bool(canonical), f"canonical downstream rows={len(canonical)}")
        composition_set = set(composition_index)
        _check(checks, "combined_subset_of_composition", set(canonical) <= composition_set, f"excluded_post_fovs={sorted(composition_set - set(canonical))}")
        for raw_name in ("enrichment", "niche_gene"):
            if raw_name in table_indices:
                raw_set = set(table_indices[raw_name])
                comparison = _index_comparison(canonical, table_indices[raw_name], raw_name)
                missing = comparison["missing"]
                covers = set(canonical) <= raw_set
                if raw_name == "enrichment" and missing:
                    for field_of_view in missing:
                        cell_count = fov_lookup.loc[field_of_view, "cell_count"] if field_of_view in fov_lookup.index else None
                        if cell_count is not None and int(cell_count) <= 1:
                            structurally_zero_enrichment_ids.append(field_of_view)
                        else:
                            structurally_zero_enrichment_fatal_ids.append(field_of_view)
                    enrichment_allowed = not structurally_zero_enrichment_fatal_ids
                    _check(checks, "enrichment_structural_zero_missing", enrichment_allowed, f"allowed_singleton_ids={structurally_zero_enrichment_ids} fatal_ids={structurally_zero_enrichment_fatal_ids}")
                    covers = enrichment_allowed
                elif raw_name == "enrichment":
                    _check(checks, "enrichment_structural_zero_missing", True, "no canonical enrichment rows are missing")
                _check(checks, f"{raw_name}_contains_canonical", covers, f"missing={missing} extras={comparison['extra']}")
                _check(checks, f"combined_subset_of_{raw_name}", covers, f"canonical combined rows are covered by {raw_name}")
                _check(checks, f"{raw_name}_producer_order", comparison["order_equal"], "producer order is recorded only; order mismatch is allowed", required=False)
            else:
                if raw_name == "enrichment":
                    _check(checks, "enrichment_structural_zero_missing", False, "enrichment table is missing")
                _check(checks, f"{raw_name}_contains_canonical", False, f"{raw_name} table is missing")
                _check(checks, f"combined_subset_of_{raw_name}", False, f"{raw_name} table is missing")
        _reconcile_composition(checks, table_frames["combined"], composition.reindex(canonical))
        _check(checks, "combined_index_is_canonical", table_indices["combined"] == canonical, "combined row order defines the frozen downstream index")
    else:
        _check(checks, "combined_canonical_index", False, "combined feature table is missing or unreadable")
        for raw_name in ("enrichment", "niche_gene"):
            _check(checks, f"{raw_name}_contains_canonical", False, "canonical combined index is unavailable")
            _check(checks, f"combined_subset_of_{raw_name}", False, "canonical combined index is unavailable")

    # Raw tables may have extras and producer-specific row order. Their only
    # required index relation is complete coverage of the frozen combined index.
    index_audit: list[dict[str, Any]] = []
    if composition_index:
        index_audit.append(_index_comparison(composition_index, canonical, "composition_source_to_combined"))
    for name in ("enrichment", "niche_gene"):
        if name in table_indices:
            index_audit.append(_index_comparison(canonical, table_indices[name], name))
    if canonical:
        index_audit.append(_index_comparison(canonical, canonical, "combined_canonical"))

    for name, value_column in (("targets", "target"), ("groups", "group")):
        if name not in table_frames:
            continue
        observed = table_indices[name]
        exact_index = observed == canonical
        _check(checks, f"{name}_index_exact", exact_index, f"canonical order={canonical}; observed={observed}")
        try:
            values = _table_values(table_frames[name], name)
            if name == "targets":
                expected_values = [str(fov_lookup.loc[key, "label"]) for key in canonical]
            else:
                expected_values = [str(fov_lookup.loc[key, "patient"]) for key in canonical]
            exact_values = exact_index and values == expected_values
            _check(checks, f"{name}_exact_values", exact_values, f"{value_column} values match canonical FOV metadata")
        except (ContractAuditError, KeyError) as exc:
            _check(checks, f"{name}_exact_values", False, str(exc))

    fov_frame = pd.DataFrame(fov_rows).set_index("field_of_view") if fov_rows else pd.DataFrame()
    canonical_frame = fov_frame.reindex(canonical) if not fov_frame.empty else fov_frame
    counts = {
        "cell_rows": len(post_fov),
        "composition_fov_rows": len(composition_index),
        "canonical_fov_rows": len(canonical),
        "excluded_post_fov_rows": len(set(composition_index) - set(canonical)),
        "all_post_fov_patients": int(fov_frame["patient"].nunique()) if not fov_frame.empty else 0,
        "all_post_fov_classes": int(fov_frame["label"].nunique()) if not fov_frame.empty else 0,
        "all_post_fov_class_counts": fov_frame["label"].value_counts(sort=False).to_dict() if not fov_frame.empty else {},
        "all_post_fov_patient_counts": fov_frame["patient"].value_counts(sort=False).to_dict() if not fov_frame.empty else {},
        "canonical_patients": int(canonical_frame["patient"].nunique()) if not canonical_frame.empty else 0,
        "canonical_classes": int(canonical_frame["label"].nunique()) if not canonical_frame.empty else 0,
        "canonical_class_counts": canonical_frame["label"].value_counts(sort=False).to_dict() if not canonical_frame.empty else {},
        "canonical_patient_counts": canonical_frame["patient"].value_counts(sort=False).to_dict() if not canonical_frame.empty else {},
        "patients": int(fov_frame["patient"].nunique()) if not fov_frame.empty else 0,
        "classes": int(fov_frame["label"].nunique()) if not fov_frame.empty else 0,
        "class_counts": fov_frame["label"].value_counts(sort=False).to_dict() if not fov_frame.empty else {},
        "patient_counts": fov_frame["patient"].value_counts(sort=False).to_dict() if not fov_frame.empty else {},
        "cell_class_counts": pd.Series(post_disease).value_counts(sort=False).to_dict() if post_disease else {},
        "cell_patient_counts": pd.Series(post_patient).value_counts(sort=False).to_dict() if post_patient else {},
    }
    required_checks = [item for item in checks if item["required"]]
    accepted = bool(required_checks) and all(item["passed"] for item in required_checks)
    return {
        "name": candidate.name,
        "source_h5ad": str(candidate.source_h5ad),
        "run_output": str(candidate.run_output),
        "artifact_presence_distinct_from_acceptance": True,
        "artifacts": inventory,
        "checks": checks,
        "accepted": accepted,
        "eligible_frozen_comparison_contract": accepted,
        "counts": counts,
        "resolved_keys": resolved_keys,
        "feature_columns": feature_columns,
        "composition_columns": [str(column) for column in composition.columns],
        "composition_proportions": composition.to_dict(orient="index"),
        "row_intersections": row_intersections,
        "canonical_index": canonical,
        "excluded_post_fov_ids": sorted(set(composition_index) - set(canonical)),
        "structurally_zero_enrichment_fov_ids": structurally_zero_enrichment_ids,
        "structurally_zero_enrichment_fov_count": len(structurally_zero_enrichment_ids),
        "structurally_zero_enrichment_fatal_fov_ids": structurally_zero_enrichment_fatal_ids,
        "enrichment_zero_fill_policy": {
            "formula_derived": True,
            "formula": "log2((0+1)/(0+1))=0",
            "not_label_imputation": True,
            "reason": "missing enrichment rows are allowed only for FOVs with at most one post-NMF observation",
        },
        "fov_rows": fov_rows,
        "index_comparisons": index_audit,
        "errors": errors,
        "decision_policy": "prefer valid completed historical_164; use fullsweep_225 only if independently complete and valid; never use classifier performance",
    }


def _comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {result["name"]: result for result in results}
    historical = by_name.get("historical_164", {})
    fullsweep = by_name.get("fullsweep_225", {})
    left = [str(value) for value in historical.get("canonical_index", [])]
    right = [str(value) for value in fullsweep.get("canonical_index", [])]
    left_set, right_set = set(left), set(right)
    return {
        "historical_fov_count": len(left),
        "fullsweep_fov_count": len(right),
        "intersection_count": len(left_set & right_set),
        "historical_only": sorted(left_set - right_set),
        "fullsweep_only": sorted(right_set - left_set),
        "difference_accounting": "Exact FOV set/index intersections only. No outcome-driven filter or biological reason is inferred from the 164/225 difference.",
        "observable_difference_reasons": {
            "row_count_delta": len(right) - len(left),
            "historical_only_fov_ids": sorted(left_set - right_set),
            "fullsweep_only_fov_ids": sorted(right_set - left_set),
            "not_inferred": "The audit records presence, set/order, labels, patients, and counts; it does not attribute the difference to an outcome-driven filter.",
        },
        "eligible_candidates": [result["name"] for result in results if result["eligible_frozen_comparison_contract"]],
        "selected_policy_candidate": "historical_164" if historical.get("eligible_frozen_comparison_contract") else ("fullsweep_225" if fullsweep.get("eligible_frozen_comparison_contract") else None),
    }


def run_audit(candidates: Iterable[Candidate] = DEFAULT_CANDIDATES, output_dir: str | Path = "runs/novae_downstream_contract_audit") -> Path:
    """Publish a complete audit atomically; refuse to overwrite an output run."""
    candidates = tuple(candidates)
    output = Path(output_dir)
    if output.exists():
        raise ContractAuditError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent))
    try:
        results = [audit_candidate(candidate) for candidate in candidates]
        payload = {
            "scope": "read-only SLURM-only NOVAE downstream skin pseudo-FOV/NMF contract audit",
            "required_contract": list(REQUIRED_ARTIFACTS),
            "candidates": results,
            "comparison": _comparison(results),
            "acceptance": "all required checks must pass; no classifier-performance criterion is used",
            "read_only": True,
            "no_real_local_h5ad": True,
            "outputs": ["novae_downstream_contract_audit.json", "contract_checks.csv", "artifact_inventory.csv", "index_audit.csv"],
        }
        check_rows = []
        artifact_rows = []
        index_rows = []
        for result in results:
            check_rows.extend({"candidate": result["name"], **check} for check in result["checks"])
            artifact_rows.extend({"candidate": result["name"], **item} for item in result["artifacts"].values())
            index_rows.extend({"candidate": result["name"], **row} for row in result["index_comparisons"])
        _atomic_json(payload, staging / "novae_downstream_contract_audit.json")
        _atomic_csv(pd.DataFrame(check_rows), staging / "contract_checks.csv")
        _atomic_csv(pd.DataFrame(artifact_rows), staging / "artifact_inventory.csv")
        _atomic_csv(pd.DataFrame(index_rows), staging / "index_audit.csv")
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-source", type=Path, default=Path(HISTORICAL_SOURCE))
    parser.add_argument("--historical-run-output", type=Path, default=Path(HISTORICAL_RUN))
    parser.add_argument("--fullsweep-source", type=Path, default=Path(FULLSWEEP_SOURCE))
    parser.add_argument("--fullsweep-run-output", type=Path, default=Path(FULLSWEEP_RUN))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run_audit(
            (
                Candidate("historical_164", args.historical_source, args.historical_run_output),
                Candidate("fullsweep_225", args.fullsweep_source, args.fullsweep_run_output),
            ),
            args.output_dir,
        )
    except (ContractAuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
