#!/usr/bin/env python3
"""Read-only comparison of baseline and explicit-scale NOVAE runs.

This CPU utility opens H5ADs backed and never touches ``adata.X``.  It compares
row/variable identity, canonical graph topology, validity/coverage, latent
agreement, resolution assignments, and resolved FIDE/JSD provenance.  Inputs
are never rewritten; reports are published atomically.
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
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

GRAPH_KEY = "spatial_connectivities"
VALID_KEY = "neighborhood_valid"
DOMAIN_PREFIX = "novae_domains_res"
DEFAULT_COVERAGE = 0.70


class NovaComparisonError(ValueError):
    """Comparison input or output contract violation."""


def _atomic_json(payload: Any, path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        os.close(fd)
        frame.to_csv(name, index=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise NovaComparisonError(f"resolved manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NovaComparisonError(f"could not read resolved manifest: {path}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("run"), Mapping):
        raise NovaComparisonError(f"resolved manifest must contain a run object: {path}")
    return dict(payload)


def _find_artifact(run_dir: Path, suffix: str) -> Path:
    candidates = sorted(run_dir.glob(f"*{suffix}"))
    if len(candidates) != 1:
        raise NovaComparisonError(f"expected exactly one *{suffix} in {run_dir}, found {len(candidates)}")
    return candidates[0]


def _resolve_side(h5ad: Path | None, manifest: Path | None, run_dir: Path | None) -> tuple[Path, Path]:
    if run_dir is not None:
        if h5ad is None:
            h5ad = _find_artifact(run_dir, "_zero_shot.h5ad")
        if manifest is None:
            manifests = sorted(run_dir.glob("*resolved_manifest*.json"))
            if len(manifests) != 1:
                manifests = sorted(run_dir.glob("*provenance*.json"))
            if len(manifests) != 1:
                raise NovaComparisonError(f"expected one resolved/provenance manifest in {run_dir}, found {len(manifests)}")
            manifest = manifests[0]
    if h5ad is None or manifest is None:
        raise NovaComparisonError("each side requires --*-h5ad and --*-manifest, or --*-run-dir")
    if not h5ad.is_file():
        raise NovaComparisonError(f"H5AD does not exist: {h5ad}")
    return h5ad, manifest


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _assigned(value: Any) -> bool:
    if _missing(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _bool_mask(adata: Any) -> np.ndarray:
    if VALID_KEY not in adata.obs:
        raise NovaComparisonError(f"obs[{VALID_KEY!r}] is missing")
    values = adata.obs[VALID_KEY].tolist()
    if any(_missing(value) or not isinstance(value, (bool, np.bool_)) for value in values):
        raise NovaComparisonError(f"obs[{VALID_KEY!r}] must contain only booleans")
    return np.asarray(values, dtype=bool)


def _graph_signature(adata: Any) -> dict[str, Any]:
    if GRAPH_KEY not in adata.obsp:
        raise NovaComparisonError(f"obsp[{GRAPH_KEY!r}] is missing")
    graph = adata.obsp[GRAPH_KEY]
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)
    graph = graph.tocsr(copy=True)
    graph.sort_indices()
    graph.eliminate_zeros()
    rows, cols = graph.nonzero()
    pairs = np.column_stack((rows.astype(np.int64), cols.astype(np.int64)))
    digest = hashlib.sha256(pairs.tobytes()).hexdigest()
    return {"shape": [int(x) for x in graph.shape], "directed_edges": int(len(pairs)),
            "sha256": digest, "pairs": pairs}


def _coverage(adata: Any, valid: np.ndarray, slide_key: str, domain_columns: list[str], side: str) -> tuple[list[dict[str, Any]], bool]:
    if slide_key not in adata.obs:
        raise NovaComparisonError(f"obs[{slide_key!r}] is missing")
    slides = np.asarray([str(x) for x in adata.obs[slide_key].tolist()])
    rows: list[dict[str, Any]] = []
    no_valid_missing = True
    for domain in domain_columns:
        labels = adata.obs[domain].tolist()
        assigned = np.asarray([_assigned(value) for value in labels], dtype=bool)
        no_valid_missing &= not bool(np.any(valid & ~assigned))
        for slide in sorted(set(slides)):
            mask = slides == slide
            total = int(mask.sum())
            count = int(assigned[mask].sum())
            valid_count = int(valid[mask].sum())
            rows.append({"side": side, "domain_key": domain, "slide": slide, "total": total,
                         "valid_neighborhood": valid_count, "assigned": count,
                         "unassigned": total - count, "coverage": count / total if total else 0.0,
                         "valid_assignment_missing": int(np.count_nonzero(valid[mask] & ~assigned[mask]))})
        total = len(assigned)
        rows.append({"side": side, "domain_key": domain, "slide": "__overall__", "total": int(total),
                     "valid_neighborhood": int(valid.sum()), "assigned": int(assigned.sum()),
                     "unassigned": int(total - assigned.sum()),
                     "coverage": float(assigned.mean()) if total else 0.0,
                     "valid_assignment_missing": int(np.count_nonzero(valid & ~assigned))})
    return rows, no_valid_missing


def _finite_latent(adata: Any, key: str) -> tuple[np.ndarray, np.ndarray]:
    if key not in adata.obsm:
        raise NovaComparisonError(f"latent key {key!r} is missing")
    values = np.asarray(adata.obsm[key], dtype=float)
    if values.ndim != 2 or values.shape[0] != adata.n_obs:
        raise NovaComparisonError("latent representation is not two-dimensional and row aligned")
    finite = np.isfinite(values).all(axis=1)
    return values, finite


def _latent_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    l2 = np.linalg.norm(a[mask] - b[mask], axis=1) if np.any(mask) else np.array([])
    norms = np.linalg.norm(a[mask], axis=1) * np.linalg.norm(b[mask], axis=1) if np.any(mask) else np.array([])
    dot = np.sum(a[mask] * b[mask], axis=1) if np.any(mask) else np.array([])
    cosine = dot[norms > 0] / norms[norms > 0] if len(norms) else np.array([])
    return {
        "common_valid_finite_rows": int(mask.sum()),
        "direct_l2_mean": float(np.mean(l2)) if len(l2) else np.nan,
        "direct_l2_median": float(np.median(l2)) if len(l2) else np.nan,
        "direct_l2_max": float(np.max(l2)) if len(l2) else np.nan,
        "cosine_mean": float(np.mean(cosine)) if len(cosine) else np.nan,
        "cosine_median": float(np.median(cosine)) if len(cosine) else np.nan,
        "cosine_min": float(np.min(cosine)) if len(cosine) else np.nan,
        "metrics_finite": bool(len(l2) and np.isfinite(l2).all() and len(cosine) and np.isfinite(cosine).all()),
    }


def _latent_comparison(left: Any, right: Any, left_key: str, right_key: str,
                      baseline_valid: np.ndarray, sensitivity_valid: np.ndarray,
                      slide_key: str) -> dict[str, Any]:
    a, af = _finite_latent(left, left_key)
    b, bf = _finite_latent(right, right_key)
    common = baseline_valid & sensitivity_valid & af & bf
    summary = {
        "baseline_key": left_key, "sensitivity_key": right_key,
        "baseline_dimension": int(a.shape[1]), "sensitivity_dimension": int(b.shape[1]),
        "dimension_match": bool(a.shape[1] == b.shape[1]),
        "baseline_finite_rows": int(af.sum()), "sensitivity_finite_rows": int(bf.sum()),
        "baseline_valid_rows": int(baseline_valid.sum()), "sensitivity_valid_rows": int(sensitivity_valid.sum()),
        "common_valid_finite_rows": int(common.sum()), "finite_rows_match": bool(np.array_equal(af, bf)),
        **_latent_stats(a, b, common),
    }
    if slide_key not in left.obs or slide_key not in right.obs:
        raise NovaComparisonError(f"obs[{slide_key!r}] is missing for latent per-slide comparison")
    left_slides = np.asarray([str(x) for x in left.obs[slide_key].tolist()])
    right_slides = np.asarray([str(x) for x in right.obs[slide_key].tolist()])
    if not np.array_equal(left_slides, right_slides):
        raise NovaComparisonError("baseline and sensitivity slide values are not exactly aligned")
    per_slide = []
    for slide in sorted(set(left_slides)):
        mask = common & (left_slides == slide)
        row = {"slide": slide, **_latent_stats(a, b, mask)}
        row["baseline_rows"] = int(np.count_nonzero(left_slides == slide))
        row["sensitivity_rows"] = int(np.count_nonzero(right_slides == slide))
        row["baseline_valid_rows"] = int(np.count_nonzero(baseline_valid & (left_slides == slide)))
        row["sensitivity_valid_rows"] = int(np.count_nonzero(sensitivity_valid & (right_slides == slide)))
        per_slide.append(row)
    summary["per_slide"] = per_slide
    summary["per_slide_metrics_finite"] = bool(per_slide and all(row["metrics_finite"] for row in per_slide))
    return summary


def _resolution_columns(adata: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in adata.obs.columns:
        if not str(column).startswith(DOMAIN_PREFIX):
            continue
        token = str(column)[len(DOMAIN_PREFIX):].lstrip("_")
        result[token] = str(column)
    return result


def _ari_nmi(left: list[Any], right: list[Any]) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        return np.nan, np.nan
    a = [str(x) for x in left]; b = [str(x) for x in right]
    if len(set(a)) == 1 and len(set(b)) == 1:
        return 1.0, 1.0
    table = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy(dtype=float)
    n = float(table.sum())
    choose = lambda x: x * (x - 1.0) / 2.0
    index = float(sum(choose(x) for x in table.ravel()))
    expected = sum(choose(x) for x in table.sum(axis=1)) * sum(choose(x) for x in table.sum(axis=0)) / choose(n) if n > 1 else 0.0
    maximum = 0.5 * (sum(choose(x) for x in table.sum(axis=1)) + sum(choose(x) for x in table.sum(axis=0)))
    ari = (index - expected) / (maximum - expected) if maximum != expected else 1.0
    p = table / n
    pi = p.sum(axis=1); pj = p.sum(axis=0)
    nz = p > 0
    mi = float(sum(p[i, j] * np.log(p[i, j] / (pi[i] * pj[j])) for i, j in zip(*np.where(nz))))
    hi = float(-sum(x * np.log(x) for x in pi if x > 0)); hj = float(-sum(x * np.log(x) for x in pj if x > 0))
    nmi = 2 * mi / (hi + hj) if hi + hj else 1.0
    return float(ari), float(nmi)


def _domain_comparison(left: Any, right: Any, valid: np.ndarray) -> list[dict[str, Any]]:
    left_cols, right_cols = _resolution_columns(left), _resolution_columns(right)
    rows: list[dict[str, Any]] = []
    for token in sorted(set(left_cols) | set(right_cols)):
        if token not in left_cols or token not in right_cols:
            rows.append({"resolution": token, "available": False, "reason": "resolution missing on one side"})
            continue
        lk, rk = left_cols[token], right_cols[token]
        la = np.asarray([_assigned(x) for x in left.obs[lk].tolist()]); ra = np.asarray([_assigned(x) for x in right.obs[rk].tolist()])
        common = valid & la & ra
        left_values = left.obs[lk].tolist()
        right_values = right.obs[rk].tolist()
        common_indices = np.flatnonzero(common)
        ari, nmi = _ari_nmi([left_values[i] for i in common_indices],
                             [right_values[i] for i in common_indices])
        left_sizes = {str(label): int(sum(1 for value in left_values if _assigned(value) and str(value) == str(label))) for label in sorted({str(value) for value in left_values if _assigned(value)})}
        right_sizes = {str(label): int(sum(1 for value in right_values if _assigned(value) and str(value) == str(label))) for label in sorted({str(value) for value in right_values if _assigned(value)})}
        rows.append({"resolution": token, "baseline_domain_key": lk, "sensitivity_domain_key": rk,
                     "available": bool(common.any()), "common_valid_assigned": int(common.sum()),
                     "baseline_domain_count": len(left_sizes), "sensitivity_domain_count": len(right_sizes),
                     "baseline_domain_sizes": left_sizes, "sensitivity_domain_sizes": right_sizes,
                     "ari": ari, "nmi": nmi,
                     "metrics_finite": bool(np.isfinite(ari) and np.isfinite(nmi))})
    return rows


def _run_object(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    run = payload.get("run", payload)
    if not isinstance(run, Mapping):
        raise NovaComparisonError("manifest run object is not a mapping")
    return run


def _resolution_keys(run: Mapping[str, Any], fallback: Mapping[str, Any]) -> set[str]:
    requested = run.get("requested_resolutions")
    if isinstance(requested, Mapping):
        return {str(key) for key in requested}
    return {str(key) for key in fallback}


def _science_rows(left_manifest: Mapping[str, Any], right_manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool, bool]:
    left_run, right_run = _run_object(left_manifest), _run_object(right_manifest)
    a = left_run.get("science_metrics", {})
    b = right_run.get("science_metrics", {})
    if not isinstance(a, Mapping): a = {}
    if not isinstance(b, Mapping): b = {}
    expected_a = _resolution_keys(left_run, a)
    expected_b = _resolution_keys(right_run, b)
    expected_identical = expected_a == expected_b
    expected = expected_a | expected_b | {str(key) for key in a} | {str(key) for key in b}
    rows: list[dict[str, Any]] = []
    for token in sorted(expected):
        av, bv = a.get(token), b.get(token)
        def metric(obj: Any, name: str) -> float:
            try: return float(obj[name])
            except (TypeError, KeyError, ValueError): return np.nan
        fide_a, fide_b = metric(av, "FIDE"), metric(bv, "FIDE")
        jsd_a, jsd_b = metric(av, "JSD"), metric(bv, "JSD")
        rows.append({"resolution": token, "expected_baseline": token in expected_a,
                     "expected_sensitivity": token in expected_b,
                     "baseline_FIDE": fide_a, "sensitivity_FIDE": fide_b,
                     "baseline_JSD": jsd_a, "sensitivity_JSD": jsd_b,
                     "available": bool(token in expected_a and token in expected_b and av is not None and bv is not None),
                     "metrics_finite": bool(np.isfinite([fide_a, fide_b, jsd_a, jsd_b]).all())})
    all_pairs = bool(rows and all(row["available"] and row["metrics_finite"] for row in rows))
    return rows, expected_identical, all_pairs


def _fixed_design(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def value(run: Mapping[str, Any], key: str) -> Any:
        return run.get(key)
    checks: dict[str, bool] = {}
    checks["input_sha256_identical"] = bool(value(left, "input_sha256") and value(left, "input_sha256") == value(right, "input_sha256"))
    checks["checkpoint_sha256_identical"] = bool(value(left, "checkpoint_sha256") and value(left, "checkpoint_sha256") == value(right, "checkpoint_sha256"))
    checks["model_revision_request_identical"] = bool(value(left, "model_revision") and value(left, "model_revision") == value(right, "model_revision"))
    checks["seed_identical"] = bool(value(left, "seed") is not None and value(left, "seed") == value(right, "seed"))
    checks["expression_mode_identical"] = bool(value(left, "expression_mode") and value(left, "expression_mode") == value(right, "expression_mode"))
    checks["requested_resolutions_identical"] = bool(value(left, "requested_resolutions") and value(left, "requested_resolutions") == value(right, "requested_resolutions"))
    checks["primary_resolution_identical"] = bool(value(left, "primary_resolution") is not None and value(left, "primary_resolution") == value(right, "primary_resolution"))
    checks["technology_identical"] = bool(value(left, "technology") and value(left, "technology") == value(right, "technology"))
    checks["distance_qc_expected_um_predeclared"] = bool(value(left, "distance_qc_expected_um") == 100.0 and value(right, "distance_qc_expected_um") == 100.0)
    checks["distance_qc_tolerance_predeclared"] = bool(value(left, "distance_qc_relative_tolerance") == 0.5 and value(right, "distance_qc_relative_tolerance") == 0.5)
    checks["minimum_assignment_coverage_predeclared"] = bool(value(left, "minimum_domain_assignment_coverage") == 0.70 and value(right, "minimum_domain_assignment_coverage") == 0.70)
    checks["slide_key_predeclared"] = bool(value(left, "slide_key") == "sample_id" and value(right, "slide_key") == "sample_id")
    checks["group_key_predeclared"] = bool(value(left, "group_key") == "patient" and value(right, "group_key") == "patient")
    checks["reference_identical_all"] = bool(value(left, "reference") == "all" and value(right, "reference") == "all")
    checks["inference_mode_identical_zero_shot"] = bool(value(left, "inference_mode") == "zero_shot" and value(right, "inference_mode") == "zero_shot")
    checks["baseline_coordinate_strategy"] = value(left, "coordinate_strategy") == "visium_manifest"
    checks["sensitivity_coordinate_strategy"] = value(right, "coordinate_strategy") == "visium_explicit_scale"
    baseline_radius = left.get("radius_pruning")
    sensitivity_radius = right.get("radius_pruning")
    checks["baseline_radius_pruning_removed_zero_edges"] = bool(isinstance(baseline_radius, Mapping) and baseline_radius.get("applied") is True and baseline_radius.get("removed_undirected_edges") == 0)
    checks["sensitivity_radius_pruning_omitted"] = sensitivity_radius is None or (isinstance(sensitivity_radius, Mapping) and sensitivity_radius.get("applied") is False)
    return {**checks, "fixed_design_contract": bool(all(checks.values()))}


def compare_runs(baseline_h5ad: str | Path, sensitivity_h5ad: str | Path,
                 baseline_manifest: str | Path, sensitivity_manifest: str | Path,
                 output_dir: str | Path, *, slide_key: str = "sample_id",
                 minimum_coverage: float = DEFAULT_COVERAGE) -> Path:
    """Compare two runs and atomically publish JSON/CSV reports."""
    output = Path(output_dir)
    if output.exists():
        raise NovaComparisonError(f"refusing existing output directory: {output}")
    if not np.isfinite(float(minimum_coverage)) or float(minimum_coverage) != DEFAULT_COVERAGE:
        raise NovaComparisonError("minimum coverage is fixed at the predeclared 0.70 protocol")
    if slide_key != "sample_id":
        raise NovaComparisonError("slide_key is fixed at the predeclared sample_id protocol")
    baseline_h5ad, sensitivity_h5ad = Path(baseline_h5ad), Path(sensitivity_h5ad)
    bm, sm = _read_manifest(Path(baseline_manifest)), _read_manifest(Path(sensitivity_manifest))
    parent = output.parent; parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=parent))
    try:
        try:
            import anndata as ad
        except ImportError as exc:
            raise RuntimeError("anndata is required for NOVAE comparison") from exc
        left, right = ad.read_h5ad(baseline_h5ad, backed="r"), ad.read_h5ad(sensitivity_h5ad, backed="r")
        left_ids, right_ids = [str(x) for x in left.obs_names], [str(x) for x in right.obs_names]
        left_vars, right_vars = [str(x) for x in left.var_names], [str(x) for x in right.var_names]
        alignment = {"obs_names_exact": left_ids == right_ids, "var_names_exact": left_vars == right_vars,
                     "obs_count_baseline": len(left_ids), "obs_count_sensitivity": len(right_ids),
                     "var_count_baseline": len(left_vars), "var_count_sensitivity": len(right_vars)}
        if not alignment["obs_names_exact"] or not alignment["var_names_exact"]:
            raise NovaComparisonError("baseline and sensitivity obs_names/var_names are not exactly aligned")
        lv, rv = _bool_mask(left), _bool_mask(right)
        valid_exact = bool(np.array_equal(lv, rv))
        graph_a, graph_b = _graph_signature(left), _graph_signature(right)
        graph_set_a = {tuple(pair) for pair in graph_a["pairs"].tolist()}
        graph_set_b = {tuple(pair) for pair in graph_b["pairs"].tolist()}
        graph_only_baseline = sorted(graph_set_a - graph_set_b)
        graph_only_sensitivity = sorted(graph_set_b - graph_set_a)
        graph_edge_diff_count = len(graph_only_baseline) + len(graph_only_sensitivity)
        graph_identity = bool(graph_a["shape"] == graph_b["shape"] and not graph_edge_diff_count)
        graph_rows = [{"side": "baseline", "shape": graph_a["shape"], "directed_edges": graph_a["directed_edges"], "baseline_only_edges": len(graph_only_baseline), "sensitivity_only_edges": len(graph_only_sensitivity), "edge_diff_count": graph_edge_diff_count, "sha256": graph_a["sha256"]},
                      {"side": "sensitivity", "shape": graph_b["shape"], "directed_edges": graph_b["directed_edges"], "baseline_only_edges": len(graph_only_baseline), "sensitivity_only_edges": len(graph_only_sensitivity), "edge_diff_count": graph_edge_diff_count, "sha256": graph_b["sha256"]}]
        graph_rows.append({"side": "comparison", "shape": graph_a["shape"], "directed_edges": graph_a["directed_edges"], "baseline_only_edges": len(graph_only_baseline), "sensitivity_only_edges": len(graph_only_sensitivity), "edge_diff_count": graph_edge_diff_count, "sha256": "identity" if graph_identity else "different"})
        left_run, right_run = bm["run"], sm["run"]
        left_key = str(left_run.get("latent_key", "novae_latent")); right_key = str(right_run.get("latent_key", "novae_latent"))
        common_valid = lv & rv
        latent = _latent_comparison(left, right, left_key, right_key, lv, rv, slide_key)
        left_domains, right_domains = _resolution_columns(left), _resolution_columns(right)
        domains = _domain_comparison(left, right, common_valid)
        coverage_a, missing_a = _coverage(left, lv, slide_key, sorted(left_domains.values()), "baseline")
        coverage_b, missing_b = _coverage(right, rv, slide_key, sorted(right_domains.values()), "sensitivity")
        coverage = coverage_a + coverage_b
        coverage_pass = bool(coverage and all(float(row["coverage"]) >= float(minimum_coverage) for row in coverage))
        science, science_keys_identical, science_pairs_available_finite = _science_rows(bm, sm)
        fixed_design = _fixed_design(bm["run"], sm["run"])
        domain_available = bool(domains and all(row.get("available", False) for row in domains))
        comparisons_available = bool(latent["common_valid_finite_rows"] > 0 and latent["dimension_match"] and domain_available and science_pairs_available_finite)
        finite_metrics = bool(latent["metrics_finite"] and latent["per_slide_metrics_finite"] and all(row.get("metrics_finite", False) for row in domains) and science_pairs_available_finite)
        acceptance = {
            "row_var_alignment": bool(alignment["obs_names_exact"] and alignment["var_names_exact"]),
            "graph_identity": graph_identity,
            "validity_masks_exact": valid_exact,
            "coverage_at_least_0.70_overall_and_per_slide": coverage_pass,
            "no_valid_missing_labels": bool(missing_a and missing_b),
            "finite_metrics": finite_metrics,
            "science_expected_resolution_keys_identical": science_keys_identical,
            "science_fide_jsd_pairs_available_finite": science_pairs_available_finite,
            "fixed_design_contract": fixed_design["fixed_design_contract"],
            "comparisons_available": comparisons_available,
        }
        acceptance["overall_accepted"] = bool(all(acceptance.values()))
        payload = {"scope": "read-only technical/predeclared nominal-100um protocol comparison; not biological stability",
                   "protocol": {"minimum_coverage": DEFAULT_COVERAGE, "slide_key": "sample_id",
                                "no_ari_nmi_equivalence_threshold": True},
                   "baseline_h5ad": str(baseline_h5ad.resolve()), "sensitivity_h5ad": str(sensitivity_h5ad.resolve()),
                   "baseline_manifest": str(Path(baseline_manifest).resolve()), "sensitivity_manifest": str(Path(sensitivity_manifest).resolve()),
                   "minimum_coverage": float(minimum_coverage), "acceptance": acceptance,
                   "alignment": alignment, "validity_masks_exact": valid_exact,
                   "fixed_design": fixed_design,
                   "science": {"expected_resolution_keys_identical": science_keys_identical,
                               "all_fide_jsd_pairs_available_finite": science_pairs_available_finite},
                   "graph": {"identity": graph_identity, "baseline_sha256": graph_a["sha256"], "sensitivity_sha256": graph_b["sha256"],
                             "baseline_edges": graph_a["directed_edges"], "sensitivity_edges": graph_b["directed_edges"],
                             "edge_diff_count": graph_edge_diff_count,
                             "baseline_only_sample": [list(pair) for pair in graph_only_baseline[:20]],
                             "sensitivity_only_sample": [list(pair) for pair in graph_only_sensitivity[:20]]},
                   "latent": latent, "domain_resolution": domains, "domain_resolution_count": len(domains), "science_metric_count": len(science),
                   "note": "Expression X was not loaded or accessed; inputs were opened backed read-only and not mutated."}
        _atomic_json(payload, staging / "novae_comparison.json")
        _atomic_csv(pd.DataFrame(graph_rows), staging / "graph_comparison.csv")
        _atomic_csv(pd.DataFrame(coverage), staging / "coverage_comparison.csv")
        _atomic_csv(pd.DataFrame(coverage)[["side", "domain_key", "slide", "total", "assigned", "unassigned", "valid_assignment_missing"]], staging / "assignment_missingness.csv")
        _atomic_csv(pd.DataFrame([latent]), staging / "latent_comparison.csv")
        _atomic_csv(pd.DataFrame(latent["per_slide"]), staging / "latent_per_slide_comparison.csv")
        _atomic_csv(pd.DataFrame(domains), staging / "domain_comparison.csv")
        _atomic_csv(pd.DataFrame(science), staging / "science_metrics_comparison.csv")
        staging.replace(output)
    except BaseException:
        try: left.file.close(); right.file.close()
        except Exception: pass
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try: left.file.close(); right.file.close()
    except Exception: pass
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("baseline", "sensitivity"):
        parser.add_argument(f"--{prefix}-h5ad", type=Path)
        parser.add_argument(f"--{prefix}-manifest", type=Path)
        parser.add_argument(f"--{prefix}-run-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--slide-key", default="sample_id")
    parser.add_argument("--minimum-coverage", type=float, default=DEFAULT_COVERAGE)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        bh, bm = _resolve_side(args.baseline_h5ad, args.baseline_manifest, args.baseline_run_dir)
        sh, sm = _resolve_side(args.sensitivity_h5ad, args.sensitivity_manifest, args.sensitivity_run_dir)
        compare_runs(bh, sh, bm, sm, args.output_dir, slide_key=args.slide_key, minimum_coverage=args.minimum_coverage)
    except (NovaComparisonError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
