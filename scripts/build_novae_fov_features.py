#!/usr/bin/env python3
"""Build the frozen historical-164 NOVAE res1.0 FOV feature branch.

This adapter is intentionally separate from the NOVAE inference pipeline.  It
reads the two real H5ADs only when invoked by the CPU SLURM job, joins cells by
``unique_cell_id``, and publishes a complete immutable FOV table.  Synthetic
unit tests may call the pure dataframe/matrix helpers directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree

HISTORICAL_BASE_H5AD = "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/cosmx_with_nmf.h5ad"
CALIBRATED_NOVAE_H5AD = "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot/paired_cpu_diagnostic/skin_visium_ssc_paired_cpu_calibrated/novae_skin_visium_ssc_paired_cpu_calibrated_zero_shot.h5ad"
HISTORICAL_FEATURE_DIR = "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_inputs"
HISTORICAL_SOURCE_OUTPUT = "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs"
DEFAULT_OUTPUT = "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_res1_fov_features_historical164"
DOMAIN_KEY = "novae_domains_res1.0"
VALIDITY_KEY = "neighborhood_valid"
DEFAULT_DOMAINS = tuple(f"L{i}" for i in range(9))
_DOMAIN_RE = re.compile(r"^L\d+$")


class ContractError(ValueError):
    """Raised when an input violates the frozen adapter contract."""


def natural_domain_order(values: Iterable[Any]) -> list[str]:
    labels = {str(v) for v in values if not _is_missing_label(v)}
    if not labels:
        raise ContractError("calibrated NOVAE has no assigned domains")
    if not all(_DOMAIN_RE.fullmatch(label) for label in labels):
        raise ContractError(f"res1.0 domain vocabulary must use L<number>: {sorted(labels)}")
    ordered = sorted(labels, key=lambda x: int(x[1:]))
    numbers = [int(label[1:]) for label in ordered]
    if numbers != list(range(numbers[-1] + 1)) or numbers[-1] > 8:
        raise ContractError(f"res1.0 domain vocabulary must be contiguous L0-L8: {ordered}")
    return ordered


def _is_missing_label(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "na", "null", "unassigned", "unknown"}:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _pick(columns: Iterable[str], names: tuple[str, ...], label: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise ContractError(f"missing {label}; accepted={names}")


def _metadata(obs: pd.DataFrame, label: str, *, require_context: bool = True, require_fov: bool = True) -> pd.DataFrame:
    frame = obs.copy()
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ContractError(f"{label} obs names are not unique")
    unique = _pick(frame.columns, ("unique_cell_id", "unique_cellid", "cell_id", "cell_ID"), f"{label} cell key")
    patient = next((name for name in ("patient", "Patient", "subject", "sample_id") if name in frame), None)
    disease = next((name for name in ("Disease_State", "disease_state", "Disease/Health State", "Disease.Health.State") if name in frame), None)
    if require_context and (patient is None or disease is None):
        raise ContractError(f"{label} is missing patient or disease metadata")
    fov = None
    if "unique_fov" in frame:
        fov = frame["unique_fov"]
    elif "field_of_view" in frame:
        fov = frame["field_of_view"]
    elif "fov" in frame and patient is not None:
        fov = frame[patient].astype(str) + "_" + frame["fov"].astype(str)
    elif require_fov:
        raise ContractError(f"{label} missing unique_fov/fov")
    checks = [("unique_cell_id", frame[unique])]
    if patient is not None:
        checks.append(("patient", frame[patient]))
    if disease is not None:
        checks.append(("Disease_State", frame[disease]))
    if fov is not None:
        checks.append(("fov", fov))
    for name, values in checks:
        if values.isna().any() or values.astype(str).str.strip().isin({"", "nan", "None"}).any():
            raise ContractError(f"{label} {name} contains missing/blank values")
    result = pd.DataFrame({"unique_cell_id": frame[unique].astype(str).to_numpy()}, index=frame.index)
    if patient is not None:
        result["patient"] = frame[patient].astype(str).to_numpy()
    if disease is not None:
        result["Disease_State"] = frame[disease].astype(str).to_numpy()
    if fov is not None:
        result["fov_key"] = fov.astype(str).to_numpy()
    if result["unique_cell_id"].duplicated().any():
        raise ContractError(f"{label} unique_cell_id is not unique")
    return result


def _read_table(stem: Path) -> pd.DataFrame:
    path = stem if stem.suffix else stem.with_suffix(".parquet")
    if not path.exists() and not stem.suffix:
        path = stem.with_suffix(".csv")
    if not path.exists():
        raise ContractError(f"missing table: {stem}")
    frame = pd.read_csv(path, index_col=0) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ContractError(f"{path} has duplicate index")
    return frame


def _same_series(expected: pd.Series, observed: pd.Series, label: str) -> None:
    if not expected.index.equals(observed.index) or not expected.astype(str).equals(observed.astype(str)):
        raise ContractError(f"{label} does not match frozen canonical index/order/values")


def _validate_fov_consistency(metadata: pd.DataFrame, label: str) -> None:
    grouped = metadata.groupby("fov_key", sort=False)
    bad = [fov for fov, frame in grouped if frame["patient"].nunique() != 1 or frame["Disease_State"].nunique() != 1]
    if bad:
        raise ContractError(f"{label} has mixed patient/disease metadata within FOV(s): {bad[:10]}")


def _parse_validity(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "valid"}:
            return True
        if token in {"false", "0", "no", "invalid"}:
            return False
    raise ContractError(f"unrecognized {VALIDITY_KEY} value: {value!r}")


def _validate_provenance(adata: Any) -> dict[str, Any]:
    provenance = getattr(adata, "uns", {}).get("novae_pilot_provenance")
    if not isinstance(provenance, dict):
        raise ContractError("NOVAE H5AD is missing uns['novae_pilot_provenance']")
    expected = {
        "analysis_scope": "exploratory", "reference": "all", "inference_mode": "zero_shot",
        "dataset_id": "skin_visium_ssc_paired_cpu_calibrated", "coordinate_strategy": "visium_explicit_scale",
        "primary_resolution": 1.0, "domain_key": DOMAIN_KEY, "neighborhood_valid_key": VALIDITY_KEY,
        "accelerator": "cpu", "device": "cpu", "workers": 0,
        "confirmatory_held_out_classification_allowed": False,
        "input_sha256": "262418e8e7ed06de805e940406f3ae9e41487ce085da1ae8f940c81f95daf6dd",
        "checkpoint_sha256": "1422f9f72d6e532921bf8a90f0996f1c46c6891f6ecbc73e404521ec5aa7b04a",
        "minimum_domain_assignment_coverage": 0.70,
    }
    for key, value in expected.items():
        if key not in provenance or provenance[key] != value:
            raise ContractError(f"NOVAE provenance mismatch for {key}: expected {value!r}")
    policy = provenance.get("deterministic_policy")
    requested = policy.get("requested") if isinstance(policy, dict) else None
    effective = policy.get("effective") if isinstance(policy, dict) else None
    if (
        not isinstance(policy, dict)
        or not isinstance(requested, (bool, np.bool_))
        or not isinstance(effective, (bool, np.bool_))
        or bool(requested) is not True
        or bool(effective) is not True
        or provenance.get("seed") != 42
    ):
        raise ContractError("NOVAE provenance is not the predeclared deterministic CPU run")
    return provenance


def validate_frozen_contract(feature_dir: Path, source_output_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    combined = _read_table(feature_dir / "combined_features_filtered.parquet")
    targets = _read_table(feature_dir / "targets_y.parquet").squeeze("columns")
    groups = _read_table(feature_dir / "groups.parquet").squeeze("columns")
    combined.index = combined.index.astype(str)
    targets.index = targets.index.astype(str)
    groups.index = groups.index.astype(str)
    if not combined.index.is_unique or not targets.index.is_unique or not groups.index.is_unique:
        raise ContractError("frozen feature/target/group indexes must be unique")
    if not combined.index.equals(targets.index) or not combined.index.equals(groups.index):
        raise ContractError("frozen combined/targets/groups indexes differ")
    if len(combined) != 164:
        raise ContractError(f"historical_164 frozen contract requires 164 FOVs, got {len(combined)}")
    if groups.astype(str).nunique() != 14:
        raise ContractError(f"historical_164 frozen contract requires 14 patient groups, got {groups.nunique()}")
    class_counts = sorted(targets.astype(str).value_counts().tolist())
    if class_counts != [61, 103]:
        raise ContractError(f"historical_164 frozen contract requires class counts [61, 103], got {class_counts}")
    if not np.isfinite(combined.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ContractError("frozen combined features contain non-finite values")
    meta_path = source_output_dir / "post_nmf_obs.csv"
    if not meta_path.exists():
        raise ContractError("frozen source output is missing post_nmf_obs.csv")
    for stem in ("enrichment_features_fov", "niche_gene_features_fov"):
        if not ((source_output_dir / f"{stem}.parquet").exists() or (source_output_dir / f"{stem}.csv").exists()):
            raise ContractError(f"frozen source output is missing {stem}")
    if meta_path.exists():
        post = pd.read_csv(meta_path, index_col=0)
        post.index = post.index.astype(str)
        meta = _metadata(post, "post_nmf_obs")
        _validate_fov_consistency(meta, "post_nmf_obs")
        fov_meta = meta.groupby("fov_key", sort=False).agg({"patient": "first", "Disease_State": "first"})
        expected_fov = pd.DataFrame({"patient": groups.astype(str), "Disease_State": targets.astype(str)})
        expected_fov.index = expected_fov.index.astype(str)
        if set(fov_meta.index) != set(expected_fov.index):
            raise ContractError("post_nmf_obs FOV set differs from frozen canonical index")
        for key in expected_fov.index:
            if str(fov_meta.loc[key, "patient"]) != str(expected_fov.loc[key, "patient"]):
                raise ContractError(f"patient conflict for frozen FOV {key}")
            if str(fov_meta.loc[key, "Disease_State"]) != str(expected_fov.loc[key, "Disease_State"]):
                raise ContractError(f"disease conflict for frozen FOV {key}")
    return combined, targets, groups


def _compute_area(obs: pd.DataFrame) -> pd.Series:
    if "Area" in obs.columns:
        area = pd.to_numeric(obs["Area"], errors="coerce")
        if area.notna().any():
            return area
    width = "Width" if "Width" in obs.columns else "width" if "width" in obs.columns else None
    height = "Height" if "Height" in obs.columns else "height" if "height" in obs.columns else None
    if width and height:
        area = pd.to_numeric(obs[width], errors="coerce") * pd.to_numeric(obs[height], errors="coerce")
        if area.notna().any():
            return area
    return pd.Series(np.full(len(obs), np.pi * (15.0 / 2.0) ** 2), index=obs.index, dtype=float)


def _domain_assignments(base_obs: pd.DataFrame, novae_obs: pd.DataFrame, domains: list[str] | None = None) -> pd.DataFrame:
    base = _metadata(base_obs, "authoritative H5AD")
    _validate_fov_consistency(base, "authoritative H5AD")
    # The calibrated NOVAE file is the original-section H5AD and need not carry
    # the post-NMF pseudo-FOV key.  Only the authoritative H5AD owns FOV mapping.
    nova = _metadata(novae_obs, "calibrated NOVAE H5AD", require_context=False, require_fov=False)
    if set(base.unique_cell_id) != set(nova.unique_cell_id):
        raise ContractError("authoritative and calibrated NOVAE unique_cell_id sets differ")
    if DOMAIN_KEY not in novae_obs or VALIDITY_KEY not in novae_obs:
        raise ContractError(f"NOVAE .obs must contain {DOMAIN_KEY!r} and {VALIDITY_KEY!r}")
    if base["unique_cell_id"].duplicated().any() or nova["unique_cell_id"].duplicated().any():
        raise ContractError("unique_cell_id must be unique in both H5ADs")
    # Join by key, deliberately not by positional order.
    out = base.set_index("unique_cell_id").copy()
    nov = novae_obs.copy()
    nov.index = nov.index.astype(str)
    nov["_key"] = nova["unique_cell_id"].to_numpy()
    nov = nov.set_index("_key")
    out["_domain"] = nov[DOMAIN_KEY].reindex(out.index).to_numpy()
    out["_valid"] = nov[VALIDITY_KEY].reindex(out.index).to_numpy()
    # Compare shared context by key, but never require pseudo-FOV metadata in NOVAE.
    nova_by_id = nova.set_index("unique_cell_id")
    for col in ("patient", "Disease_State"):
        if col in nova_by_id and not out[col].astype(str).equals(nova_by_id[col].reindex(out.index).astype(str)):
            raise ContractError(f"NOVAE/base metadata conflict for {col}")
    for candidate in ("sample", "Sample", "sample_id", "slide_id", "Slide_ID"):
        if candidate in base_obs.columns and candidate in novae_obs.columns:
            base_values = base_obs.assign(_key=base["unique_cell_id"].to_numpy()).set_index("_key")[candidate]
            nova_values = novae_obs.assign(_key=nova["unique_cell_id"].to_numpy()).set_index("_key")[candidate]
            if not base_values.reindex(out.index).astype(str).equals(nova_values.reindex(out.index).astype(str)):
                raise ContractError(f"NOVAE/base metadata conflict for {candidate}")
    valid = out["_valid"].map(_parse_validity)
    assigned = ~out["_domain"].map(_is_missing_label)
    if (valid & ~assigned).any() or ((~valid) & assigned).any():
        raise ContractError("valid rows must be labeled and invalid rows must be unlabeled")
    out["_valid"] = valid
    out["_assigned"] = valid & assigned
    found = natural_domain_order(out.loc[out["_assigned"], "_domain"])
    if domains is not None and found != list(domains):
        raise ContractError(f"domain vocabulary differs from frozen manifest: {found} != {domains}")
    out["_domain"] = out["_domain"].map(lambda x: str(x) if not _is_missing_label(x) else pd.NA)
    return out


def composition_features(assignments: pd.DataFrame, canonical_index: pd.Index, domains: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.DataFrame(0, index=canonical_index.astype(str), columns=domains, dtype=int)
    assigned = assignments[assignments["_assigned"]]
    if not assigned.empty:
        counts = counts.add(pd.crosstab(assigned["fov_key"].astype(str), assigned["_domain"]).reindex(index=counts.index, columns=domains, fill_value=0), fill_value=0).astype(int)
    props = counts.astype(float).copy()
    sums = props.sum(axis=1)
    props = props.div(sums.replace(0, np.nan), axis=0).fillna(0.0)
    props.columns = [f"novae_prop_{domain}" for domain in domains]
    return counts, props


def enrichment_features(assignments: pd.DataFrame, canonical_index: pd.Index, domains: list[str]) -> pd.DataFrame:
    """Reproduce notebook BallTree/radius/Area formula, with NOVAE labels."""
    required = {"CenterX_global_px", "CenterY_global_px", "Area"}
    missing = required - set(assignments.columns)
    if missing:
        # Caller may have supplied width/height, matching notebook _compute_area.
        if missing != {"Area"} or not ({"Width", "Height"} <= set(assignments.columns) or {"width", "height"} <= set(assignments.columns)):
            raise ContractError(f"enrichment requires notebook spatial columns: {sorted(missing)}")
    work = assignments.copy()
    work["_area"] = _compute_area(work)
    assigned_mask = work["_assigned"].to_numpy()
    assigned_coords = work.loc[assigned_mask, ["CenterX_global_px", "CenterY_global_px"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    assigned_area = pd.to_numeric(work.loc[assigned_mask, "_area"], errors="coerce").to_numpy()
    if not np.isfinite(assigned_coords).all() or not np.isfinite(assigned_area).all() or (assigned_area <= 0).any():
        raise ContractError("assigned NOVAE spots require finite coordinates and positive finite Area")
    cols = [f"enrichment_{i + 1}-{j + 1}" for i in range(len(domains)) for j in range(len(domains))]
    rows: list[dict[str, Any]] = []
    for fov, group in work[work["_assigned"]].groupby("fov_key", sort=False):
        coords = group[["CenterX_global_px", "CenterY_global_px"]].apply(pd.to_numeric, errors="coerce").to_numpy()
        diameters = 2.0 * np.sqrt(pd.to_numeric(group["_area"], errors="coerce").to_numpy() / np.pi)
        labels = group["_domain"].astype(str).to_numpy()
        matrix = pd.DataFrame(0.0, index=domains, columns=domains)
        if len(coords) >= 2:
            tree = BallTree(coords)
            for i in range(len(coords)):
                radius = 2.0 * diameters[i]
                if not np.isfinite(radius) or radius <= 0:
                    continue
                neighbors = tree.query_radius(coords[i : i + 1], r=radius)[0]
                neighbors = neighbors[neighbors != i]
                if len(neighbors):
                    for domain, count in pd.Series(labels[neighbors]).value_counts().items():
                        matrix.loc[labels[i], domain] += float(count)
        matrix = matrix + matrix.T
        proportions = pd.Series(labels).value_counts(normalize=True).reindex(domains, fill_value=0.0).to_numpy()
        total = float(matrix.to_numpy().sum())
        expected = total * np.outer(proportions, proportions)
        values = np.log2((matrix.to_numpy() + 1.0) / (expected + 1.0))
        row = {
            f"enrichment_{i + 1}-{j + 1}": float(values[i, j])
            for i in range(len(domains))
            for j in range(len(domains))
        }
        row["fov_key"] = str(fov)
        rows.append(row)
    result = pd.DataFrame(rows).set_index("fov_key") if rows else pd.DataFrame(index=pd.Index([], name="fov_key"))
    result = result.reindex(canonical_index.astype(str)).fillna(0.0)
    return result.reindex(columns=cols, fill_value=0.0).astype(float)


def choose_expression_matrix(adata: Any) -> tuple[Any, pd.Index]:
    """Match notebook _choose_expression_matrix exactly (raw takes precedence)."""
    if getattr(adata, "raw", None) is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names.astype(str))
    return adata.X, pd.Index(adata.var_names.astype(str))


def _assert_finite_matrix(matrix: Any) -> None:
    # AnnData backed sparse matrices are CSRDataset objects rather than scipy
    # sparse matrices; materialize only this required expression matrix for QC.
    if not sparse.issparse(matrix) and hasattr(matrix, "to_memory"):
        materialized = matrix.to_memory()
        if materialized is not matrix:
            _assert_finite_matrix(materialized)
            return
    if sparse.issparse(matrix):
        finite = np.isfinite(matrix.data).all()
    else:
        finite = np.isfinite(np.asarray(matrix)).all()
    if not finite:
        raise ContractError("expression matrix contains non-finite values")


def _group_mean_expression(matrix: Any, row_groups: np.ndarray, n_groups: int) -> np.ndarray:
    if sparse.issparse(matrix):
        selector = sparse.csr_matrix((np.ones(len(row_groups)), (row_groups, np.arange(len(row_groups)))), shape=(n_groups, len(row_groups)))
        grouped = (selector @ matrix.tocsr()).toarray()
    else:
        values = np.asarray(matrix)
        grouped = np.zeros((n_groups, values.shape[1]), dtype=float)
        for row, group in enumerate(row_groups):
            grouped[group] += values[row]
    counts = np.bincount(row_groups, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    return grouped / counts[:, None]


def niche_gene_features(assignments: pd.DataFrame, adata: Any, canonical_index: pd.Index, domains: list[str]) -> pd.DataFrame:
    matrix, genes = choose_expression_matrix(adata)
    _assert_finite_matrix(matrix)
    if matrix.shape[0] != len(assignments):
        raise ContractError("expression rows do not match authoritative obs rows")
    fovs = canonical_index.astype(str).tolist()
    fov_codes = pd.Categorical(assignments["fov_key"].astype(str), categories=fovs).codes
    columns: list[str] = []
    blocks: list[np.ndarray] = []
    for domain in domains:
        positions = np.flatnonzero((assignments["_assigned"].to_numpy()) & (assignments["_domain"].astype(str).to_numpy() == domain) & (fov_codes >= 0))
        block = np.zeros((len(fovs), len(genes)), dtype=float)
        if len(positions):
            block = _group_mean_expression(matrix[positions], fov_codes[positions], len(fovs))
        blocks.append(block)
        columns.extend([f"niche_{domain}_gene_{gene}" for gene in genes])
    result = pd.DataFrame(np.concatenate(blocks, axis=1) if blocks else np.zeros((len(fovs), 0)), index=canonical_index.astype(str), columns=columns)
    if not np.isfinite(result.to_numpy()).all():
        raise ContractError("niche-gene features contain non-finite values")
    return result


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path)


def _validate_output_frames(frames: dict[str, pd.DataFrame], canonical: pd.Index, props: pd.DataFrame) -> None:
    expected = canonical.astype(str)
    for name in ("novae_composition_fov", "enrichment_features_fov", "niche_gene_features_fov", "combined_features_filtered", "novae_fov_qc"):
        frame = frames[name]
        if not frame.index.equals(expected):
            raise ContractError(f"{name} index/order differs from frozen canonical index")
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if numeric.size and not np.isfinite(numeric).all():
            raise ContractError(f"{name} contains non-finite values")
    if list(frames["combined_features_filtered"].columns) != list(props.columns):
        raise ContractError("combined_features_filtered must contain exactly NOVAE composition columns")
    for name in ("targets_y", "groups"):
        if not frames[name].index.equals(expected):
            raise ContractError(f"{name} index/order differs from frozen canonical index")


def build_from_adatas(base: Any, novae: Any, feature_dir: Path, source_output_dir: Path, output_dir: Path, *, manifest_domains: list[str] | None = None, input_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """Build and atomically publish an adapter bundle from already-open H5ADs."""
    if output_dir.exists():
        raise ContractError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    combined, targets, groups = validate_frozen_contract(feature_dir, source_output_dir)
    base_obs = base.obs.copy(); novae_obs = novae.obs.copy()
    provenance = _validate_provenance(novae)
    base_meta = _metadata(base_obs, "authoritative H5AD")
    post_path = source_output_dir / "post_nmf_obs.csv"
    if post_path.exists():
        post = pd.read_csv(post_path, index_col=0)
        post_meta = _metadata(post, "post_nmf_obs")
        if set(post_meta["unique_cell_id"]) != set(base_meta["unique_cell_id"]):
            raise ContractError("post_nmf_obs and authoritative H5AD unique_cell_id sets differ")
        base_by_id = base_meta.set_index("unique_cell_id"); post_by_id = post_meta.set_index("unique_cell_id")
        for column in ("patient", "Disease_State", "fov_key"):
            if not base_by_id[column].reindex(post_by_id.index).equals(post_by_id[column]):
                raise ContractError(f"post_nmf_obs metadata conflict for {column}")
    assignments = _domain_assignments(base_obs, novae_obs, manifest_domains)
    # Spatial/morphology columns are authoritative and retained through the key-indexed frame.
    base_lookup = base_obs.copy()
    base_lookup["_key"] = base_meta["unique_cell_id"].to_numpy()
    base_lookup = base_lookup.set_index("_key")
    for col in base_obs.columns:
        if col not in assignments.columns:
            assignments[col] = base_lookup[col].reindex(assignments.index).to_numpy()
    canonical = combined.index.astype(str)
    observed_fovs = set(assignments["fov_key"].astype(str))
    if observed_fovs != set(canonical):
        raise ContractError("authoritative H5AD FOV set differs from frozen canonical index")
    expected_domains = list(manifest_domains or DEFAULT_DOMAINS)
    if expected_domains != list(DEFAULT_DOMAINS):
        raise ContractError(f"historical res1.0 adapter requires vocabulary {list(DEFAULT_DOMAINS)}")
    domains = natural_domain_order(assignments.loc[assignments["_assigned"], "_domain"])
    if domains != expected_domains:
        raise ContractError(f"observed NOVAE vocabulary differs from expected {expected_domains}: {domains}")
    counts, props = composition_features(assignments, canonical, domains)
    enrichment = enrichment_features(assignments, canonical, domains)
    niche = niche_gene_features(assignments, base, canonical, domains)
    qc = pd.DataFrame(index=canonical)
    total = assignments.groupby("fov_key").size().reindex(canonical, fill_value=0)
    valid = assignments[assignments["_valid"]].groupby("fov_key").size().reindex(canonical, fill_value=0)
    assigned = assignments[assignments["_assigned"]].groupby("fov_key").size().reindex(canonical, fill_value=0)
    qc["total_spots"] = total.astype(int); qc["valid_spots"] = valid.astype(int); qc["assigned_spots"] = assigned.astype(int)
    qc["unassigned_spots"] = (valid - assigned).astype(int); qc["invalid_spots"] = (total - valid).astype(int)
    qc["coverage"] = qc["assigned_spots"] / qc["total_spots"].replace(0, np.nan); qc["coverage"] = qc["coverage"].fillna(0.0)
    qc["zero_assigned"] = qc["assigned_spots"].eq(0)
    # Singleton/no-pair enrichment is a formula-derived zero, not imputation.
    qc["enrichment_structural_zero"] = qc["assigned_spots"].le(1)
    qc["patient"] = groups.astype(str); qc["Disease_State"] = targets.astype(str)
    # This evaluator contract intentionally contains only the additive NOVAE
    # composition branch; inherited NMF/global-selected columns are forbidden.
    combined_out = props.reindex(canonical).copy()
    obs_columns = ["patient", "Disease_State", "fov_key", "_domain", "_valid", "_assigned"]
    obs_columns.extend(column for column in ("NMF_factor", "dominant_nmf_factor") if column in assignments.columns)
    obs_out = assignments[obs_columns].copy()
    obs_out = obs_out.rename(columns={"_domain": DOMAIN_KEY, "_valid": VALIDITY_KEY, "_assigned": "novae_assigned"})
    obs_out.index.name = "unique_cell_id"
    target_frame = targets.to_frame(name=targets.name or "Disease_State")
    group_frame = groups.to_frame(name=groups.name or "patient")
    _validate_output_frames(
        {"novae_composition_fov": props, "enrichment_features_fov": enrichment, "niche_gene_features_fov": niche,
         "combined_features_filtered": combined_out, "novae_fov_qc": qc, "targets_y": target_frame, "groups": group_frame},
        canonical, props,
    )
    temp = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent if output_dir.parent.exists() else None))
    try:
        _write_frame(obs_out, temp / "novae_domains_res1.0_obs.parquet")
        _write_frame(props, temp / "novae_composition_fov.parquet"); props.to_csv(temp / "novae_composition_fov.csv")
        _write_frame(enrichment, temp / "enrichment_features_fov.parquet"); enrichment.to_csv(temp / "enrichment_features_fov.csv")
        _write_frame(niche, temp / "niche_gene_features_fov.parquet")
        _write_frame(combined_out, temp / "combined_features_filtered.parquet")
        _write_frame(target_frame, temp / "targets_y.parquet")
        _write_frame(group_frame, temp / "groups.parquet")
        _write_frame(qc, temp / "novae_fov_qc.parquet"); qc.to_csv(temp / "novae_fov_qc.csv")
        manifest = {
            "contract": "historical_164",
            "contract_audit": {"job": "43008275", "commit": "b8218ba", "status": "COMPLETED 0:0 1:47", "selection_policy": "historical_164; never performance"},
            "domain_key": DOMAIN_KEY, "validity_key": VALIDITY_KEY, "domains": domains,
            "novae_pilot_provenance": {key: provenance.get(key) for key in ("analysis_scope", "reference", "inference_mode", "dataset_id", "coordinate_strategy", "primary_resolution", "domain_key", "neighborhood_valid_key", "accelerator", "device", "workers", "seed", "deterministic_policy", "confirmatory_held_out_classification_allowed", "input_sha256", "checkpoint_sha256", "minimum_domain_assignment_coverage")},
            "warning": "exploratory reference=all; not confirmatory and not outcome-selected",
            "formula": {"enrichment": "notebook BallTree; radius=2*(2*sqrt(Area/pi)); self excluded; symmetric interaction; log2((interaction+1)/(expected+1))", "invalid": "exclude invalid spots", "zero_assigned": "preserve all-zero row"},
            "expression_source": "authoritative base cosmx_with_nmf.h5ad", "expression_mode": "raw.X if raw exists else X", "candidate_tables": "all; exploratory reference=all warning",
            "dimensions": {"fovs": len(canonical), "cells": len(assignments), "domains": len(domains), "niche_features": niche.shape[1], "enrichment_features": enrichment.shape[1]},
            "feature_names": {"composition": list(props.columns), "enrichment": list(enrichment.columns), "niche_gene": list(niche.columns)},
            "feature_counts": {"composition": props.shape[1], "enrichment": enrichment.shape[1], "niche_gene": niche.shape[1]},
            "frozen_index_sha256": hashlib.sha256("\n".join(canonical).encode()).hexdigest(),
            "coverage": {"min": float(qc.coverage.min()), "max": float(qc.coverage.max()), "zero_assigned_fovs": int(qc.zero_assigned.sum())},
            "structural_zero_fovs": qc.index[qc.enrichment_structural_zero].tolist(), "zero_assigned_fovs": qc.index[qc.zero_assigned].tolist(), "inputs": {},
        }
        paths = dict(input_paths or {})
        paths.update({"feature_dir/combined_features_filtered.parquet": feature_dir / "combined_features_filtered.parquet", "feature_dir/targets_y.parquet": feature_dir / "targets_y.parquet", "feature_dir/groups.parquet": feature_dir / "groups.parquet"})
        if (source_output_dir / "post_nmf_obs.csv").exists():
            paths["source_output_dir/post_nmf_obs.csv"] = source_output_dir / "post_nmf_obs.csv"
        for name, path in paths.items():
            manifest["inputs"][name] = {"path": str(path), "sha256": _hash(path)}
        manifest["outputs"] = {
            str(path.relative_to(temp)): {"path": str(output_dir / path.relative_to(temp)), "sha256": _hash(path)}
            for path in sorted(temp.iterdir())
            if path.name != "novae_feature_manifest.json"
        }
        (temp / "novae_feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise ContractError(f"refusing to overwrite existing output directory: {output_dir}")
        os.replace(temp, output_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5ad", default=HISTORICAL_BASE_H5AD)
    parser.add_argument("--novae-h5ad", default=CALIBRATED_NOVAE_H5AD)
    parser.add_argument("--feature-dir", default=HISTORICAL_FEATURE_DIR)
    parser.add_argument("--source-output-dir", default=HISTORICAL_SOURCE_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-domains", default="", help="Optional comma-separated manifest vocabulary, e.g. L0,L1,...")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real H5AD feature generation is restricted to a SLURM job; use synthetic helpers/tests locally")
    expected_domains = [value.strip() for value in args.expected_domains.split(",") if value.strip()] or list(DEFAULT_DOMAINS)
    expected_domains = natural_domain_order(expected_domains)
    import anndata as ad
    base = ad.read_h5ad(args.base_h5ad, backed="r")
    novae = ad.read_h5ad(args.novae_h5ad, backed="r")
    try:
        build_from_adatas(
            base,
            novae,
            Path(args.feature_dir),
            Path(args.source_output_dir),
            Path(args.output_dir),
            manifest_domains=expected_domains,
            input_paths={"authoritative_base_h5ad": Path(args.base_h5ad), "calibrated_novae_h5ad": Path(args.novae_h5ad)},
        )
    finally:
        for obj in (base, novae):
            if getattr(obj, "file", None) is not None:
                obj.file.close()


if __name__ == "__main__":
    main()
