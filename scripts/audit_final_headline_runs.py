#!/usr/bin/env python3
"""Read-only audit of the final skin and kidney analysis evidence.

This command is deliberately an evidence audit, not a pipeline runner.  It
opens real H5AD files in backed read-only mode and writes only a new,
transactional audit directory.  Run it on HPG through the companion SLURM
launcher; tests use small in-memory tables and never process real data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

MODERN_PREDICTION_COLUMNS = {
    "outer_fold", "item_id", "test_group", "true_label", "predicted_label",
    "decision_threshold", "positive_class", "positive_class_probability",
}
MODERN_PREDICTION_MARKERS = {"outer_fold", "test_group", "true_label", "predicted_label", "decision_threshold", "positive_class"}

import numpy as np
import pandas as pd


class AuditError(ValueError):
    """An input or evidence contract failed."""


PATIENT_KEYS = ("patient_id", "patient", "Patient", "sample_id", "sample")
DISEASE_KEYS = ("disease", "Disease", "Disease_State", "condition", "diagnosis", "disease_state")
FOV_KEYS = ("fov_id", "FOV_ID", "fov", "FOV", "field_of_view", "sample_id")


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _text(value: Any, name: str) -> str:
    if _missing(value) or not str(value).strip() or str(value).strip().lower() == "nan":
        raise AuditError(f"{name} contains missing values")
    return str(value).strip()


def _pick(columns: Iterable[str], requested: str | None, candidates: Iterable[str], label: str) -> str:
    available = set(columns)
    if requested:
        if requested not in available:
            raise AuditError(f"{label} metadata column {requested!r} is missing")
        return requested
    for key in candidates:
        if key in available:
            return key
    raise AuditError(f"could not discover {label} metadata column; tried {list(candidates)}")


def validate_source_metadata(
    obs: pd.DataFrame | Mapping[str, Iterable[Any]],
    *,
    expected_patients: int,
    patient_key: str | None = None,
    disease_key: str | None = None,
    fov_key: str | None = None,
    label: str = "source",
) -> dict[str, Any]:
    """Validate patient/disease/FOV metadata without changing the table."""
    frame = obs if isinstance(obs, pd.DataFrame) else pd.DataFrame(obs)
    patient = _pick(frame.columns, patient_key, PATIENT_KEYS, f"{label} patient")
    disease = _pick(frame.columns, disease_key, DISEASE_KEYS, f"{label} disease")
    fov = _pick(frame.columns, fov_key, FOV_KEYS, f"{label} FOV")
    patients = [_text(value, f"{label}[{patient}]") for value in frame[patient].tolist()]
    diseases = [_text(value, f"{label}[{disease}]") for value in frame[disease].tolist()]
    fovs = [_text(value, f"{label}[{fov}]") for value in frame[fov].tolist()]
    patient_to_disease: dict[str, set[str]] = {}
    for p, d in zip(patients, diseases):
        patient_to_disease.setdefault(p, set()).add(d)
    inconsistent = {p: sorted(values) for p, values in patient_to_disease.items() if len(values) != 1}
    if inconsistent:
        raise AuditError(f"{label} patient→disease is not single-valued: {inconsistent}")
    observed = len(patient_to_disease)
    if observed != expected_patients:
        raise AuditError(f"{label} expected {expected_patients} patients, observed {observed}")
    fovs_by_patient = {p: len({f for pp, f in zip(patients, fovs) if pp == p}) for p in sorted(patient_to_disease)}
    return {
        "patient_key": patient,
        "disease_key": disease,
        "fov_key": fov,
        "n_obs": int(len(frame)),
        "n_patients": observed,
        "expected_patients": int(expected_patients),
        # CosMX FOV labels are often reused within each patient; count the
        # patient/FOV unit rather than collapsing identically named FOVs.
        "n_fovs": len(set(zip(patients, fovs))),
        "fovs_by_patient": fovs_by_patient,
        "patients": sorted(patient_to_disease),
        "disease_by_patient": {p: next(iter(values)) for p, values in sorted(patient_to_disease.items())},
        "disease_counts": dict(pd.Series(diseases).value_counts().sort_index()),
    }


def inspect_predictions(
    frame: pd.DataFrame,
    *,
    item_key: str | None = None,
    group_key: str | None = None,
    fold_key: str | None = None,
    true_key: str | None = None,
    probability_key: str | None = None,
    prediction_key: str | None = None,
    threshold: float = 0.5,
    positive_class: str | None = None,
) -> dict[str, Any]:
    """Check fold predictions and recompute pooled binary metrics."""
    if frame.empty:
        raise AuditError("fold_predictions is empty")
    def choose(requested: str | None, names: tuple[str, ...], label: str) -> str:
        if requested:
            if requested not in frame.columns:
                raise AuditError(f"fold_predictions {label} column {requested!r} is missing")
            return requested
        for name in names:
            if name in frame.columns:
                return name
        raise AuditError(f"could not discover fold_predictions {label} column")
    item = choose(item_key, ("item_id", "fov_id", "fov", "sample_id", "id"), "item")
    group = choose(group_key, ("test_group", "group", "group_id", "patient_id", "patient", "sample_id"), "group")
    fold = choose(fold_key, ("outer_fold", "fold", "fold_id", "test_fold"), "fold")
    truth = choose(true_key, ("true_label", "y_true", "true", "target", "label", "disease"), "true-label")
    probability = choose(probability_key, ("positive_class_probability", "probability", "prob", "y_prob", "y_score"), "probability")
    prediction = choose(prediction_key, ("predicted_label", "y_pred", "prediction", "predicted", "pred"), "prediction")
    positive_key = choose(None, ("positive_class",), "positive-class")
    threshold_key = choose(None, ("decision_threshold", "threshold"), "decision-threshold")
    items = [_text(value, "item IDs") for value in frame[item].tolist()]
    if len(items) != len(set(items)):
        raise AuditError("fold_predictions item IDs are not unique")
    folds = frame[fold].tolist()
    group_values = [_text(value, "prediction groups") for value in frame[group].tolist()]
    true_values = [_text(value, "true labels") for value in frame[truth].tolist()]
    observed = [str(value).strip() for value in frame[prediction].tolist()]
    classes = sorted(set(true_values) | set(observed))
    if len(classes) != 2:
        raise AuditError(f"prediction labels must be exactly binary, observed {classes}")
    group_folds: dict[str, set[str]] = {}
    group_labels: dict[str, set[str]] = {}
    for g, f, y in zip(group_values, folds, true_values):
        group_folds.setdefault(g, set()).add(_text(f, "fold IDs"))
        group_labels.setdefault(g, set()).add(y)
    bad_folds = {g: sorted(v) for g, v in group_folds.items() if len(v) != 1}
    bad_labels = {g: sorted(v) for g, v in group_labels.items() if len(v) != 1}
    if bad_folds:
        raise AuditError(f"each test group must be confined to one fold: {bad_folds}")
    if bad_labels:
        raise AuditError(f"group-label consistency failed: {bad_labels}")
    try:
        probs = pd.to_numeric(frame[probability], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise AuditError("probabilities must be numeric") from exc
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise AuditError("probabilities must be finite and in [0,1]")
    row_thresholds = np.full(len(frame), threshold, dtype=float)
    if threshold_key is not None:
        try:
            row_thresholds = pd.to_numeric(frame[threshold_key], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise AuditError("decision thresholds must be numeric") from exc
        if not np.isfinite(row_thresholds).all() or np.any((row_thresholds < 0) | (row_thresholds > 1)):
            raise AuditError("decision thresholds must be finite and in [0,1]")
        threshold = float(row_thresholds[0])
    if not math.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise AuditError("decision thresholds must be finite and in [0,1]")
    row_positive_classes = [_text(value, "positive_class") for value in frame[positive_key].tolist()]
    distinct_positive = sorted(set(row_positive_classes))
    if len(distinct_positive) != 1:
        raise AuditError(f"positive_class must be single-valued, observed {distinct_positive}")
    if positive_class is not None and positive_class != distinct_positive[0]:
        raise AuditError("positive_class argument disagrees with fold_predictions")
    positive_class = distinct_positive[0]
    if positive_class not in classes:
        raise AuditError(f"declared positive_class {positive_class!r} is absent from prediction labels {classes}")
    expected = [positive_class if value >= row_thresholds[index] else (next((x for x in classes if x != positive_class), "0")) for index, value in enumerate(probs)]
    mismatches = [index for index, (a, b) in enumerate(zip(observed, expected)) if a != b]
    if mismatches:
        raise AuditError(f"prediction/threshold inconsistency in rows {mismatches[:10]}")
    negative = next((x for x in classes if x != positive_class), "0")
    y_true = np.asarray([1 if value == positive_class else 0 for value in true_values], dtype=int)
    y_pred = np.asarray([1 if value == positive_class else 0 for value in observed], dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    support = {negative: int(np.sum(y_true == 0)), positive_class: int(np.sum(y_true == 1))}
    per_f1 = {}
    for klass, k in ((negative, 0), (positive_class, 1)):
        tp_k = int(np.sum((y_true == k) & (y_pred == k)))
        fp_k = int(np.sum((y_true != k) & (y_pred == k)))
        fn_k = int(np.sum((y_true == k) & (y_pred != k)))
        precision = tp_k / (tp_k + fp_k) if tp_k + fp_k else 0.0
        recall = tp_k / (tp_k + fn_k) if tp_k + fn_k else 0.0
        per_f1[klass] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_f1 = float(np.mean(list(per_f1.values())))
    weighted_f1 = float(sum(per_f1[k] * support[k] for k in per_f1) / len(y_true)) if len(y_true) else 0.0
    recalls = [tp / (tp + fn) if tp + fn else 0.0, tn / (tn + fp) if tn + fp else 0.0]
    return {
        "columns": {"item": item, "group": group, "fold": fold, "true": truth, "probability": probability, "prediction": prediction, "positive_class": positive_key, "threshold": threshold_key},
        "n_items": len(frame), "n_groups": len(group_folds), "n_folds": len(set(str(v) for v in folds)),
        "positive_class": positive_class, "threshold": threshold,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)), "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "group_folds": {g: sorted(v)[0] for g, v in sorted(group_folds.items())},
        "technical_checks": {"unique_item_ids": True, "group_single_fold": True, "group_label_consistent": True, "finite_probabilities_0_1": True, "threshold_prediction_consistent": True},
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _atomic_json(payload: Any, path: Path) -> None:
    name: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False) as handle:
            name = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def _atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    name: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False) as handle:
            name = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, default=_json_default) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        os.replace(name, path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def _parse_mlp_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {"path": str(path), "metadata_file": path.name}
    patterns = {
        "mlp_mode": r"(?im)^\s*MLP\s*mode\s*[:=]\s*(.+?)\s*$",
        "evaluation_unit": r"(?im)^\s*Evaluation\s+unit\s*[:=]\s*(.+?)\s*$",
        "outer_cv_mode": r"(?im)^\s*Outer\s+CV\s+mode\s*[:=]\s*(.+?)\s*$",
        "units": r"(?im)^\s*(?:units|n_units|number of units)\s*[:=]\s*(.+?)\s*$",
        "group_count": r"(?im)^\s*(?:group count|groups|n_groups|number of groups)\s*[:=]\s*(.+?)\s*$",
        "threshold": r"(?im)^\s*(?:threshold|decision threshold)\s*[:=]\s*([0-9.eE+-]+)",
        "positive_class": r"(?im)^\s*positive[_ ]class\s*[:=]\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value: Any = match.group(1).strip()
            if key == "threshold":
                value = float(value)
                result["decision_threshold"] = value
            result[key] = value
    fold_matches = re.findall(r"Processing\s+Outer\s+Fold\s+(\d+)\s*/\s*(\d+)", text, re.I)
    fold_ids = [int(fold) for fold, _total in fold_matches]
    declared_totals = [int(total) for _fold, total in fold_matches]
    if len(fold_ids) != len(set(fold_ids)):
        raise AuditError(f"duplicate Processing Outer Fold identities in {path}")
    if declared_totals and len(set(declared_totals)) != 1:
        raise AuditError(f"inconsistent declared outer-fold totals in {path}: {declared_totals}")
    declared_total = declared_totals[0] if declared_totals else None
    if declared_total is not None and any(fold < 1 or fold > declared_total for fold in fold_ids):
        raise AuditError(f"outer-fold identity is outside declared range in {path}")
    result["outer_fold_ids"] = sorted(fold_ids)
    result["outer_fold_count"] = len(set(fold_ids))
    result["outer_fold_declared_total"] = declared_total
    result["has_final_performance_report"] = "--- Final Performance Report ---" in text
    result["nested_cv_incomplete"] = bool(
        fold_ids and (len(set(fold_ids)) != declared_total or not result["has_final_performance_report"])
    )
    return result


def _modern_metadata_contract(metadata: dict[str, Any], path: Path) -> None:
    required = ("mlp_mode", "evaluation_unit", "outer_cv_mode")
    missing = [key for key in required if not str(metadata.get(key, "")).strip()]
    if missing:
        raise AuditError(f"modern leakage-safe fold_predictions requires metadata {missing} ({path})")


def _declared_group_count(metadata: dict[str, Any]) -> int | None:
    value = str(metadata.get("group_count", ""))
    match = re.search(r"\b(\d+)\b", value)
    return int(match.group(1)) if match else None


def _first(root: Path, names: tuple[str, ...]) -> Path | None:
    wanted = {name.lower() for name in names}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def inspect_run(run_dir: str | Path, *, name: str, expected_folds: int) -> dict[str, Any]:
    root = Path(run_dir)
    result: dict[str, Any] = {"name": name, "run_dir": str(root), "exists": root.is_dir(), "technical_artifact_consistency": True, "scientific_citability": "unknown", "artifacts": {}}
    if not root.is_dir():
        result["technical_artifact_consistency"] = False
        result["error"] = "run directory is missing"
        return result
    # Pipeline outputs commonly live in <run>/outputs while provenance lives
    # beside it in <run>/artifacts. Inspect that run family without walking
    # unrelated sibling runs.
    search_root = root.parent if root.name == "outputs" else root
    for key, names in (("manifest", ("manifest.json",)), ("run_summary", ("run_summary.json",))):
        path = _first(search_root, names)
        entry: dict[str, Any] = {"found": path is not None}
        if path:
            entry["path"] = str(path)
            try:
                entry["json"] = json.loads(path.read_text(encoding="utf-8"))
                entry["valid_json"] = True
            except (OSError, json.JSONDecodeError) as exc:
                entry.update(valid_json=False, error=str(exc)); result["technical_artifact_consistency"] = False
        result["artifacts"][key] = entry
        if not entry["found"]:
            result["technical_artifact_consistency"] = False
    post = sorted(str(path) for path in search_root.rglob("*") if path.is_file() and ("nmf" in path.name.lower() or "post_nmf" in path.name.lower()))
    result["artifacts"]["post_nmf"] = {"found": bool(post), "paths": post[:100]}
    if not post:
        result["technical_artifact_consistency"] = False
    mlp_rows = []
    for pred_path in sorted(search_root.rglob("fold_predictions.csv")):
        row: dict[str, Any] = {"path": str(pred_path), "technical_artifact_consistency": True}
        metadata_path = pred_path.parent / "mlp_results.txt"
        metadata: dict[str, Any] = _parse_mlp_metadata(metadata_path) if metadata_path.is_file() else {}
        row.update({key: value for key, value in metadata.items() if key != "path"})
        if metadata_path.is_file():
            row["metadata_path"] = str(metadata_path)
        try:
            prediction_frame = pd.read_csv(pred_path)
            columns = set(prediction_frame.columns)
            is_modern = bool(columns & MODERN_PREDICTION_MARKERS)
            row["metadata_contract"] = "modern_leakage_safe" if is_modern else "legacy_or_discontinued"
            if is_modern:
                _modern_metadata_contract(metadata, metadata_path)
                predictions = inspect_predictions(prediction_frame, threshold=float(row.get("threshold", 0.5)), positive_class=row.get("positive_class"))
                declared_groups = _declared_group_count(metadata)
                row["derived_units"] = "unique item_id"
                row["derived_item_count"] = predictions["n_items"]
                row["derived_group_count"] = predictions["n_groups"]
                if declared_groups is not None and declared_groups != predictions["n_groups"]:
                    raise AuditError(f"declared group count {declared_groups} does not match parsed test groups {predictions['n_groups']} ({metadata_path})")
                row["metrics"] = predictions
            else:
                row["note"] = "legacy/discontinued prediction artifact; modern metadata contract not applicable"
        except (OSError, ValueError, AuditError) as exc:
            row["technical_artifact_consistency"] = False; row["error"] = str(exc)
            result["technical_artifact_consistency"] = False
        mlp_rows.append(row)
    metadata_rows = []
    for metadata_path in sorted(search_root.rglob("mlp_results.txt")):
        parsed = _parse_mlp_metadata(metadata_path)
        metadata_rows.append(parsed)
    result["artifacts"]["mlp_outputs"] = mlp_rows
    result["artifacts"]["mlp_metadata"] = metadata_rows
    nested = []
    for metadata in metadata_rows:
        mode = " ".join(str(metadata.get(key, "")) for key in ("mlp_mode", "evaluation_unit", "outer_cv_mode"))
        fold_count = int(metadata.get("outer_fold_count", 0))
        if fold_count or re.search(r"nested|outer", mode, re.I):
            nested.append({
                "path": metadata["path"],
                "mlp_mode": metadata.get("mlp_mode"),
                "evaluation_unit": metadata.get("evaluation_unit"),
                "outer_cv_mode": metadata.get("outer_cv_mode"),
                "fold_count": fold_count,
                "declared_fold_count": metadata.get("outer_fold_declared_total"),
                "expected_fold_count": expected_folds,
                "final_report": bool(metadata.get("has_final_performance_report")),
                "incomplete": fold_count != (metadata.get("outer_fold_declared_total") or expected_folds) or metadata.get("outer_fold_declared_total") not in (None, expected_folds) or not bool(metadata.get("has_final_performance_report")),
            })
    result["nested_cv"] = nested
    if any(entry["incomplete"] for entry in nested):
        result["technical_artifact_consistency"] = False
    lower_names = " ".join(str(path).lower() for path in search_root.rglob("*"))
    if "discontinued" in lower_names:
        result["scientific_citability"] = "non-citable: discontinued artifact"
    elif "kidney" in name.lower():
        result["scientific_citability"] = "non-citable: n=6 and chance-separability concern"
    elif "eval_fixed" in lower_names or "tune_once" in lower_names:
        result["scientific_citability"] = "exploratory: selection-biased evaluate_fixed/tune_once"
    else:
        result["scientific_citability"] = "requires protocol review"
    if "rcausal" in lower_names:
        result["rcausal_mgm_citability"] = "exploratory: FOV/patient pseudoreplication"
    return result


PROTOCOL_EXPECTED: dict[str, dict[str, Any]] = {
    "skin_sgkf3": {"folds": 3, "accuracy": 0.530488, "balanced_accuracy": 0.499205},
    "skin": {"folds": 14, "accuracy": 0.634146, "macro_f1": 0.608467, "balanced_accuracy": 0.608467},
    "kidney": {"folds": 6, "accuracy": 1.0},
}


def _protocol_expectation(path: Path) -> tuple[str, dict[str, Any]] | None:
    lower = str(path).lower()
    if "protocolcomparison_skin_sgkf3" in lower:
        return "skin_sgkf3", PROTOCOL_EXPECTED["skin_sgkf3"]
    if "protocolcomparison_skin" in lower:
        return "skin", PROTOCOL_EXPECTED["skin"]
    if "protocolcomparison_kidney" in lower:
        return "kidney", PROTOCOL_EXPECTED["kidney"]
    return None


def _protocol_rows(run_roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in run_roots:
        for path in sorted(root.rglob("*.json")) if root.is_dir() else []:
            if "protocolcomparison" not in str(path).lower() and path.name.lower() != "protocol_comparison.json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AuditError(f"invalid protocol comparison JSON {path}: {exc}") from exc
            honest = payload.get("honest", {}) if isinstance(payload, dict) else {}
            leaky = payload.get("leaky", {}) if isinstance(payload, dict) else {}
            folds = honest.get("folds", [])
            row: dict[str, Any] = {"path": str(path), "status": "read", "honest_fold_count": len(folds) if isinstance(folds, list) else None, "honest_accuracy": honest.get("pooled_accuracy"), "honest_balanced_accuracy": honest.get("pooled_balanced_accuracy"), "honest_macro_f1": honest.get("pooled_macro_f1"), "honest_weighted_f1": honest.get("pooled_weighted_f1"), "leaky_reported_score": leaky.get("reported_best_score"), "leaky_pooled_weighted_f1": leaky.get("rerun_pooled_weighted_f1"), "scientific_citability": "honest metrics are protocol evidence; leaky metrics are selection-biased"}
            expectation = _protocol_expectation(path)
            if expectation is not None:
                label, expected = expectation
                mismatches = []
                for key, expected_value in expected.items():
                    actual_key = "honest_fold_count" if key == "folds" else f"honest_{key}"
                    actual = row.get(actual_key)
                    if actual is None or not math.isclose(float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-6):
                        mismatches.append(f"{actual_key}={actual!r}, expected {expected_value!r}")
                row["expected_evidence_label"] = label
                row["expected_evidence_verified"] = not mismatches
                if mismatches:
                    raise AuditError(f"protocol evidence mismatch for {label} ({path}): {'; '.join(mismatches)}")
            rows.append(row)
    return rows


def inspect_h5ad(path: str | Path, *, label: str, expected_patients: int | None = None, patient_key: str | None = None, disease_key: str | None = None, fov_key: str | None = None, require_metadata: bool = True) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise AuditError(f"{label} H5AD does not exist: {source}")
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anndata is required for real H5AD audits") from exc
    data = ad.read_h5ad(source, backed="r")
    try:
        result: dict[str, Any] = {"label": label, "path": str(source.resolve()), "shape": [int(data.n_obs), int(data.n_vars)], "backed_read_only": True, "obs_columns": list(data.obs.columns)}
        if data.n_obs <= 0 or data.n_vars <= 0:
            raise AuditError(f"{label} H5AD has non-positive shape")
        if expected_patients is not None or require_metadata:
            result["metadata"] = validate_source_metadata(data.obs, expected_patients=expected_patients or 0, patient_key=patient_key, disease_key=disease_key, fov_key=fov_key, label=label)
        return result
    finally:
        data.file.close()


def run_audit(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    if output.exists():
        raise AuditError(f"refusing existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent))
    try:
        sources = {
            "skin": inspect_h5ad(args.skin_h5ad, label="skin", expected_patients=14, patient_key=args.skin_patient_key, disease_key=args.skin_disease_key, fov_key=args.skin_fov_key),
            "kidney_spatial": inspect_h5ad(args.kidney_spatial_h5ad, label="kidney_spatial", expected_patients=6, patient_key=args.kidney_patient_key, disease_key=args.kidney_disease_key, fov_key=args.kidney_fov_key),
            "kidney_reference": inspect_h5ad(args.kidney_reference_h5ad, label="kidney_reference", require_metadata=False),
        }
        runs = {
            "skin_1mm_split": inspect_run(args.skin_split_run_dir, name="skin_1mm_split", expected_folds=14),
            "skin_1000_fullsweep": inspect_run(args.skin_1000_run_dir, name="skin_1000_fullsweep", expected_folds=14),
            "skin_750_fullsweep": inspect_run(args.skin_750_run_dir, name="skin_750_fullsweep", expected_folds=14),
            "skin_500_fullsweep": inspect_run(args.skin_500_run_dir, name="skin_500_fullsweep", expected_folds=14),
            "kidney_poisson75": inspect_run(args.kidney_run_dir, name="kidney_poisson75", expected_folds=6),
        }
        protocols = _protocol_rows([Path(value["run_dir"]) for value in runs.values()])
        fold_prediction_rows = []
        for run_name, run in runs.items():
            for output in run.get("artifacts", {}).get("mlp_outputs", []):
                if "metrics" in output:
                    fold_prediction_rows.append({"run": run_name, "path": output["path"], **output["metrics"]})
        payload = {
            "scope": "read-only final/headline skin and kidney run audit",
            "technical_artifact_consistency_is_distinct_from_scientific_citability": True,
            "sources": sources, "runs": runs, "protocol_comparisons": protocols,
            "fold_prediction_metrics": fold_prediction_rows,
            "protocol_json_checks": {"expected_values_checked": bool(protocols), "all_checked_values_match": all(row.get("expected_evidence_verified", True) for row in protocols)},
            "documented_evidence_register": {
                "skin_1mm_honest_logo": {"pooled_accuracy": 0.634146, "macro_f1": 0.608467, "balanced_accuracy": 0.608467},
                "skin_1mm_sgkf3": {"pooled_accuracy": 0.530488, "balanced_accuracy": 0.499205},
                "kidney_honest_nested": {"pooled_accuracy": 1.0, "scientific_citability": "non-citable: n=6/chance separability"},
                "fullsweep_selection_biased": {"1000_macro_f1": 0.44, "750_macro_f1": 0.61, "500_macro_f1": 0.59},
                "original_nested_completed_folds": {"1000": "8/14", "750": "5/14", "500": "9/14"},
                "note": "These documented values are not treated as observed verification unless the corresponding ProtocolComparison JSON is present and passes the checks above.",
            },
            "caveat": "No input H5AD or existing run artifact is modified. evaluate_fixed/tune_once outputs are exploratory selection-biased; discontinued files are non-citable; RCausalMGM disease edges are exploratory because FOVs pseudoreplicate patients.",
        }
        _atomic_json(payload, staging / "final_headline_runs_audit.json")
        source_rows = [{"dataset": key, "shape": value["shape"], "n_patients": value.get("metadata", {}).get("n_patients"), "n_fovs": value.get("metadata", {}).get("n_fovs"), "backed_read_only": value["backed_read_only"]} for key, value in sources.items()]
        run_rows = [{"name": value["name"], "run_dir": value["run_dir"], "exists": value["exists"], "technical_artifact_consistency": value["technical_artifact_consistency"], "scientific_citability": value["scientific_citability"], "nested_cv": value.get("nested_cv")} for value in runs.values()]
        _atomic_csv(source_rows, staging / "source_summary.csv"); _atomic_csv(run_rows, staging / "run_summary.csv"); _atomic_csv(protocols, staging / "protocol_summary.csv"); _atomic_csv(fold_prediction_rows, staging / "fold_prediction_summary.csv")
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = "/blue/kejun.huang/vasco.hinostroza"
    parser.add_argument("--skin-h5ad", "--skin-source-h5ad", dest="skin_h5ad", type=Path, default=Path(root + "/data/skin_dataset/processed/skin_visium_ssc_1mmfov_spatial.h5ad"))
    parser.add_argument("--kidney-spatial-h5ad", type=Path, default=Path(root + "/data/kidney_dataset/processed/kidney_cosmx_six_sample_spatial.h5ad"))
    parser.add_argument("--kidney-reference-h5ad", type=Path, default=Path(root + "/data/kidney_dataset/processed/gse183277_kidney_reference.h5ad"))
    runs = root + "/nicherunner/src/sptx-tool/runs"
    parser.add_argument("--skin-split-run-dir", "--skin-1mm-run-dir", type=Path, default=Path(runs + "/skin_visium_ssc_1mmfov_poisson75_split/outputs"))
    parser.add_argument("--skin-1000-run-dir", "--skin-1000-fullsweep-run-dir", type=Path, default=Path(runs + "/skin_visium_ssc_1000umfov_poisson75_fullsweep/outputs"))
    parser.add_argument("--skin-750-run-dir", "--skin-750-fullsweep-run-dir", type=Path, default=Path(runs + "/skin_visium_ssc_750umfov_poisson75_fullsweep/outputs"))
    parser.add_argument("--skin-500-run-dir", "--skin-500-fullsweep-run-dir", type=Path, default=Path(runs + "/skin_visium_ssc_500umfov_poisson75_fullsweep/outputs"))
    parser.add_argument("--kidney-run-dir", type=Path, default=Path(runs + "/kidney_cosmx_ssc_poisson75/outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    for prefix in ("skin", "kidney"):
        parser.add_argument(f"--{prefix}-patient-key"); parser.add_argument(f"--{prefix}-disease-key"); parser.add_argument(f"--{prefix}-fov-key")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        run_audit(build_parser().parse_args(argv))
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
