#!/usr/bin/env python3
"""Pool completed historical-164 held-out FOV predictions at patient level.

This script is intentionally a reporting operation: it never trains a model and
never reads feature tables.  Only immutable manifests and ``fold_predictions.csv``
files are consumed.  The three pooling rules are declared before any results are
read and are not selected using performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from scripts import run_novae_classifier_ablation as ablation

FULL_PRIMARY_DEFAULT = Path("/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_nmf_comparison_20260922T214308Z_1310236/historical164_nmf_vs_novae")
ABLATION_DEFAULT = Path("/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_classifier_ablation_20260923T010701Z_3910105/historical164_classifier_ablation")
CONFIGURATIONS = ("full", "composition_only", "composition_enrichment", "composition_niche")
ARMS = ("nmf", "novae")
METHODS = ("primary_mean", "sensitivity_median", "sensitivity_majority_vote")
LABELS = ("healthy", "systemic_sclerosis")
POSITIVE = LABELS[1]
THRESHOLD = 0.5
REQUIRED_COLUMNS = {"item_id", "outer_fold", "test_group", "true_label", "predicted_label", "decision_threshold", "positive_class", "positive_class_probability"}


class ContractError(ValueError):
    """Raised when an immutable source does not satisfy the frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing immutable manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"manifest must be an object: {path}")
    return value


def _manifest_inventory(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("output_sha256", "output_hashes", "outputs"):
        value = manifest.get(key)
        if isinstance(value, dict) and value:
            return value
    return None


def validate_manifest_inventory(manifest_path: Path, root: Path) -> dict[str, str]:
    """Verify a manifest's source-output hash inventory and return it."""
    manifest = _json(manifest_path)
    inventory = _manifest_inventory(manifest)
    if inventory is None:
        raise ContractError(f"{manifest_path} has no output hash inventory")
    result: dict[str, str] = {}
    excludes = {str(x) for x in manifest.get("output_sha256_excludes", []) if isinstance(x, str)}
    for relative, entry in inventory.items():
        if not isinstance(relative, str):
            raise ContractError(f"invalid output inventory path in {manifest_path}: {relative!r}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContractError(f"output inventory path escapes manifest root: {manifest_path} -> {relative!r}")
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        if not isinstance(expected, str) or len(expected) != 64:
            raise ContractError(f"invalid output hash in {manifest_path}: {relative!r}")
        if relative in excludes or Path(relative).name == manifest_path.name and Path(relative).parent == Path("."):
            continue
        root_resolved = root.resolve(strict=False)
        path = root / relative_path
        try:
            path.resolve(strict=False).relative_to(root_resolved)
        except ValueError as exc:
            raise ContractError(f"output inventory path escapes manifest root: {manifest_path} -> {relative!r}") from exc
        if not path.is_file() or sha256(path) != expected:
            raise ContractError(f"output hash mismatch: {manifest_path} -> {relative}")
        result[relative] = expected
    if not result:
        raise ContractError(f"empty output hash inventory: {manifest_path}")
    return result


def _validate_source_relationships(full_primary_dir: Path, ablation_root: Path, manifests: list[Path]) -> dict[str, Any]:
    """Validate the run/config/aggregate manifest graph published by ablation."""
    full_manifest_path, aggregate_manifest_path = manifests[:2]
    full_manifest = _json(full_manifest_path)
    aggregate = _json(aggregate_manifest_path)
    expected_protocols = {name: ablation.protocol_for(name) for name in CONFIGURATIONS}
    expected_primary_protocol = ablation.primary.PROTOCOL
    if full_manifest.get("protocol") != expected_primary_protocol:
        raise ContractError("full primary manifest protocol differs from the frozen primary protocol")
    if aggregate.get("protocol") != expected_protocols:
        raise ContractError("aggregate manifest does not contain the exact four predeclared protocols")
    expected_order = list(CONFIGURATIONS)
    if aggregate.get("completed_ablations") != expected_order:
        raise ContractError("aggregate manifest configuration order/list is not the predeclared four-config list")
    declared_output = aggregate.get("full_primary_output")
    if not isinstance(declared_output, str) or Path(declared_output).expanduser().resolve(strict=False) != full_primary_dir.resolve(strict=False):
        raise ContractError("aggregate manifest full_primary_output differs from the supplied immutable full primary")
    if aggregate.get("full_primary_unchanged") is not True:
        raise ContractError("aggregate manifest does not declare the full primary unchanged")
    actual_full_manifest_sha = sha256(full_manifest_path)
    if aggregate.get("full_primary_manifest_sha256") != actual_full_manifest_sha:
        raise ContractError("aggregate manifest full primary manifest hash does not match")
    expected_primary_code = ablation._primary_code_hashes()
    if full_manifest.get("code_sha256") != expected_primary_code:
        raise ContractError("full primary code hashes differ from the current immutable primary code")
    expected_aggregate_code = {"evaluator": sha256(ablation.EVALUATOR), "orchestrator": sha256(REPO / "scripts" / "run_novae_classifier_ablation.py"), "launcher": sha256(REPO / "scripts" / "submit_novae_classifier_ablation.sh")}
    if aggregate.get("code_sha256") != expected_aggregate_code:
        raise ContractError("aggregate classifier-ablation code hashes differ from the known runner")
    if aggregate.get("inputs") != full_manifest.get("input_sha256"):
        raise ContractError("aggregate and full primary input provenance differ")
    expected_inputs = aggregate.get("inputs")
    if not isinstance(expected_inputs, dict):
        raise ContractError("aggregate manifest is missing typed input provenance")
    for configuration, manifest_path in zip(CONFIGURATIONS[1:], manifests[2:], strict=True):
        config = _json(manifest_path)
        expected_environment = {"nmf": ablation.expected_environment(configuration, "nmf_prop_"), "novae": ablation.expected_environment(configuration, "novae_prop_")}
        if config.get("ablation") != configuration or config.get("protocol") != expected_protocols[configuration]:
            raise ContractError(f"{configuration} manifest protocol/identity differs from the predeclared protocol")
        if config.get("inputs") != expected_inputs:
            raise ContractError(f"{configuration} manifest input provenance differs from aggregate")
        if config.get("environment") != expected_environment:
            raise ContractError(f"{configuration} manifest environment differs from the frozen top-family configuration")
        if config.get("code_sha256") != expected_aggregate_code:
            raise ContractError(f"{configuration} manifest code hashes differ from aggregate")
    return aggregate


def _manifest_path(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise ContractError(f"missing manifest under {root}; expected one of {names}")


def _normalise_prediction(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ContractError(f"{source} missing columns: {sorted(missing)}")
    if len(frame) != 164:
        raise ContractError(f"{source} must contain exactly 164 predictions")
    frame = frame.copy()
    for column in ("item_id", "test_group", "true_label", "predicted_label", "positive_class"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().isin({"", "nan", "none", "null"}).any():
            raise ContractError(f"{source} has blank values in {column}")
    if frame["item_id"].duplicated().any() or frame["item_id"].isna().any():
        raise ContractError(f"{source} has duplicate or missing item_id values")
    try:
        frame["outer_fold"] = pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)
        frame["decision_threshold"] = pd.to_numeric(frame["decision_threshold"], errors="raise").astype(float)
        frame["positive_class_probability"] = pd.to_numeric(frame["positive_class_probability"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{source} has non-numeric fold/threshold/probability") from exc
    if set(frame.outer_fold) != set(range(1, 15)):
        raise ContractError(f"{source} must contain outer folds 1..14")
    if not np.isfinite(frame.positive_class_probability).all() or not frame.positive_class_probability.between(0.0, 1.0).all():
        raise ContractError(f"{source} probabilities must be finite and in [0,1]")
    if not np.isfinite(frame.decision_threshold).all() or not np.isclose(frame.decision_threshold, THRESHOLD).all():
        raise ContractError(f"{source} thresholds must all equal 0.5")
    if set(frame.true_label) != set(LABELS) or frame.true_label.value_counts().to_dict() != {"healthy": 61, "systemic_sclerosis": 103}:
        raise ContractError(f"{source} must have healthy/systemic_sclerosis labels with 61/103 FOV counts")
    if set(frame.predicted_label) - set(LABELS):
        raise ContractError(f"{source} contains an unknown predicted label")
    if set(frame.positive_class) != {POSITIVE}:
        raise ContractError(f"{source} positive class must be systemic_sclerosis")
    expected_predicted = np.where(frame.positive_class_probability >= THRESHOLD, POSITIVE, LABELS[0])
    if not np.array_equal(frame.predicted_label.to_numpy(), expected_predicted):
        raise ContractError(f"{source} predicted labels are inconsistent with probability and fixed 0.5 threshold")
    patient_labels = frame.groupby("test_group", sort=False).true_label.nunique()
    patient_true = frame.groupby("test_group", sort=False).true_label.first()
    if len(patient_labels) != 14 or not patient_labels.eq(1).all() or patient_true.value_counts().to_dict() != {"healthy": 4, "systemic_sclerosis": 10}:
        raise ContractError(f"{source} must contain 4 healthy and 10 systemic_sclerosis patients with one true class per patient")
    patient_folds = frame.groupby("test_group", sort=False).outer_fold.nunique()
    if not patient_folds.eq(1).all():
        raise ContractError(f"{source} patients must be entirely in one outer fold")
    fold_patients = frame.groupby("outer_fold", sort=True).test_group.nunique()
    if len(fold_patients) != 14 or not fold_patients.eq(1).all():
        raise ContractError(f"{source} must map exactly one patient to each outer fold")
    return frame


def validate_prediction_alignment(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Validate exact item/order and metadata alignment across all eight inputs."""
    if set(frames) != {(configuration, arm) for configuration in CONFIGURATIONS for arm in ARMS}:
        raise ContractError("exactly four configurations and two arms are required")
    reference: pd.DataFrame | None = None
    for key in (("full", "nmf"), ("full", "novae"), *( (configuration, arm) for configuration in CONFIGURATIONS[1:] for arm in ARMS)):
        frame = frames[key]
        if reference is None:
            reference = frame
            continue
        for column in ("item_id", "true_label", "test_group", "outer_fold", "decision_threshold", "positive_class"):
            left = reference[column].to_numpy()
            right = frame[column].to_numpy()
            if not np.array_equal(left, right):
                raise ContractError(f"alignment mismatch in {key[0]}/{key[1]} column {column}")
    assert reference is not None
    return reference


def pool_patient_predictions(frame: pd.DataFrame, configuration: str, arm: str) -> pd.DataFrame:
    """Apply the frozen three methods to one validated FOV prediction table."""
    rows: list[dict[str, Any]] = []
    for patient, group in frame.groupby("test_group", sort=True):
        probabilities = group.positive_class_probability.to_numpy(dtype=float)
        votes = (group.predicted_label == POSITIVE).to_numpy()
        true_label = str(group.true_label.iloc[0])
        fold = int(group.outer_fold.iloc[0])
        mean_score = float(probabilities.mean())
        median_score = float(np.median(probabilities))
        vote_fraction = float(votes.mean())
        tie = bool(votes.sum() * 2 == len(votes))
        decisions = {
            "primary_mean": (mean_score, POSITIVE if mean_score >= THRESHOLD else LABELS[0], False),
            "sensitivity_median": (median_score, POSITIVE if median_score >= THRESHOLD else LABELS[0], False),
            "sensitivity_majority_vote": (vote_fraction, POSITIVE if (votes.sum() * 2 > len(votes) or (tie and mean_score >= THRESHOLD)) else LABELS[0], tie),
        }
        for method, (score, predicted, method_tie) in decisions.items():
            rows.append({"configuration": configuration, "arm": arm, "method": method, "patient": str(patient), "outer_fold": fold, "fov_count": len(group), "score": score, "vote_fraction": vote_fraction, "tie": method_tie, "true_label": true_label, "predicted_label": predicted, "correct": bool(predicted == true_label)})
    result = pd.DataFrame(rows)
    if len(result) != 14 * len(METHODS):
        raise ContractError("patient pooling did not produce exactly 14 rows per method")
    return result


def patient_metrics(patient_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute long metrics and confusion counts from patient predictions."""
    metric_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for (configuration, arm, method), group in patient_predictions.groupby(["configuration", "arm", "method"], sort=False):
        y_true, y_pred = group.true_label, group.predicted_label
        matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
        tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
        values = {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)), "weighted_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0))}
        for metric, value in values.items():
            metric_rows.append({"configuration": configuration, "arm": arm, "method": method, "metric": metric, "value": value, "n_patients": len(group), "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        matrix_rows.append({"configuration": configuration, "arm": arm, "method": method, "healthy_as_healthy": tn, "healthy_as_systemic_sclerosis": fp, "systemic_sclerosis_as_healthy": fn, "systemic_sclerosis_as_systemic_sclerosis": tp})
    return pd.DataFrame(metric_rows), pd.DataFrame(matrix_rows)


def _input_paths(full_primary_dir: Path, ablation_root: Path) -> tuple[dict[tuple[str, str], Path], list[Path]]:
    paths: dict[tuple[str, str], Path] = {}
    manifests: list[Path] = []
    full_manifest = _manifest_path(full_primary_dir, "run_manifest.json", "manifest.json")
    manifests.append(full_manifest)
    for arm in ARMS:
        paths["full", arm] = full_primary_dir / arm / "fold_predictions.csv"
    aggregate_manifest = _manifest_path(ablation_root / "aggregate", "run_manifest.json", "aggregate_manifest.json")
    manifests.append(aggregate_manifest)
    for configuration in CONFIGURATIONS[1:]:
        config_dir = ablation_root / configuration
        config_manifest = _manifest_path(config_dir, "config_manifest.json", "run_manifest.json")
        manifests.append(config_manifest)
        for arm in ARMS:
            paths[configuration, arm] = config_dir / arm / "fold_predictions.csv"
    return paths, manifests


def _validate_sources(full_primary_dir: Path, ablation_root: Path) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, Any], list[Path]]:
    paths, manifests = _input_paths(full_primary_dir, ablation_root)
    # Validate the declared configuration list before consuming predictions.
    aggregate = _validate_source_relationships(full_primary_dir, ablation_root, manifests)
    for manifest in manifests:
        # Inventories are relative to the directory containing their manifest;
        # this is also how the existing full/config/aggregate writers publish them.
        validate_manifest_inventory(manifest, manifest.parent)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise ContractError(f"missing immutable fold predictions: {path}")
        frames[key] = _normalise_prediction(pd.read_csv(path), path)
    validate_prediction_alignment(frames)
    return frames, aggregate, manifests


def _tree_hashes(root: Path, exclude: Iterable[Path] = ()) -> dict[str, str]:
    excluded = {path.resolve() for path in exclude}
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file() and path.resolve() not in excluded}


def run_pooling(*, full_primary_dir: Path, ablation_root: Path, output_root: Path) -> dict[str, Any]:
    """Validate immutable inputs and atomically publish all pooling reports."""
    full_primary_dir, ablation_root, output_root = map(Path, (full_primary_dir, ablation_root, output_root))
    if output_root.exists():
        raise ContractError(f"refusing existing output root (no overwrite): {output_root}")
    resolved_output = output_root.resolve(strict=False)
    for source in (full_primary_dir.resolve(strict=False), ablation_root.resolve(strict=False)):
        if resolved_output == source or resolved_output in source.parents or source in resolved_output.parents:
            raise ContractError("output root overlaps an input root")
    frames, aggregate, manifests = _validate_sources(full_primary_dir, ablation_root)
    before = {str(path): sha256(path) for path in [*manifests, *[path for path in _input_paths(full_primary_dir, ablation_root)[0].values()]]}
    pooled = pd.concat([pool_patient_predictions(frames[key], *key) for key in ((configuration, arm) for configuration in CONFIGURATIONS for arm in ARMS)], ignore_index=True)
    metrics, matrices = patient_metrics(pooled)
    primary = metrics[metrics.method == "primary_mean"].copy()
    summary = primary.pivot(index=["configuration", "arm"], columns="metric", values="value").reset_index()
    summary["warning"] = "n=14 patients (4 healthy, 10 systemic_sclerosis); coarse exploratory; no p-values/significance"
    paired_rows = []
    for configuration in CONFIGURATIONS:
        left = pooled[(pooled.configuration == configuration) & (pooled.arm == "nmf")].rename(columns={"correct": "nmf_correct", "predicted_label": "nmf_predicted_label", "score": "nmf_score"})
        right = pooled[(pooled.configuration == configuration) & (pooled.arm == "novae")].rename(columns={"correct": "novae_correct", "predicted_label": "novae_predicted_label", "score": "novae_score"})
        merged = left[["configuration", "method", "patient", "true_label", "nmf_correct", "nmf_predicted_label", "nmf_score"]].merge(right[["method", "patient", "novae_correct", "novae_predicted_label", "novae_score"]], on=["method", "patient"], validate="one_to_one")
        merged["correct_delta_novae_minus_nmf"] = merged.novae_correct.astype(int) - merged.nmf_correct.astype(int)
        paired_rows.append(merged)
    paired = pd.concat(paired_rows, ignore_index=True)
    descriptive = pooled.groupby(["configuration", "arm", "method"], as_index=False).agg(n_patients=("patient", "nunique"), mean_fov_count=("fov_count", "mean"), median_fov_count=("fov_count", "median"), correct_patients=("correct", "sum"))
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=parent))
    try:
        pooled.to_csv(stage / "patient_predictions_long.csv", index=False)
        metrics.to_csv(stage / "patient_metrics_long.csv", index=False)
        summary.to_csv(stage / "primary_mean_summary.csv", index=False)
        (stage / "primary_mean_summary.json").write_text(json.dumps({"rows": summary.to_dict(orient="records"), "warning": str(summary.warning.iloc[0]), "protocol": {"threshold": THRESHOLD, "positive_class": POSITIVE, "method": "arithmetic mean of held-out FOV probabilities"}}, indent=2) + "\n", encoding="utf-8")
        matrices.to_csv(stage / "confusion_matrices_long.csv", index=False)
        for _, row in matrices.iterrows():
            name = f"confusion_matrix_{row.configuration}_{row.arm}_{row.method}.csv"
            pd.DataFrame([[row.healthy_as_healthy, row.healthy_as_systemic_sclerosis], [row.systemic_sclerosis_as_healthy, row.systemic_sclerosis_as_systemic_sclerosis]], index=LABELS, columns=LABELS).to_csv(stage / name)
        paired.to_csv(stage / "paired_correctness_long.csv", index=False)
        descriptive.to_csv(stage / "configuration_descriptive.csv", index=False)
        after = {str(path): sha256(path) for path in [*manifests, *[path for path in _input_paths(full_primary_dir, ablation_root)[0].values()]]}
        if before != after:
            raise ContractError("an immutable input changed during pooling")
        manifest = {"protocol": {"cohort": "historical-164", "unit": "FOV predictions pooled to patient", "threshold": THRESHOLD, "positive_class": POSITIVE, "methods": METHODS, "configurations": CONFIGURATIONS, "arms": ARMS, "training": "none; completed immutable out-of-fold predictions only"}, "warning": "n=14 (4 healthy, 10 systemic_sclerosis), coarse exploratory, no p-values/significance; NOVAE reference=all", "inputs": {"full_primary": str(full_primary_dir), "ablation_root": str(ablation_root), "manifest_sha256": {str(path): sha256(path) for path in manifests}, "fold_predictions_sha256": {str(path): sha256(path) for path in _input_paths(full_primary_dir, ablation_root)[0].values()}}, "code_sha256": {"pooling_script": sha256(Path(__file__)), "launcher": sha256(REPO / "scripts" / "submit_novae_patient_level_pooling.sh"), "documentation": sha256(REPO / "docs" / "NOVAE_PATIENT_LEVEL_POOLING.md")}, "output_sha256": _tree_hashes(stage), "output_sha256_excludes": ["pooling_manifest.json"]}
        (stage / "pooling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output_root.exists():
            raise ContractError(f"refusing existing output root (no overwrite): {output_root}")
        os.replace(stage, output_root)
        return manifest
    except Exception:
        for path in stage.glob("*"):
            if path.is_file():
                path.unlink()
        stage.rmdir()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-primary-dir", type=Path, default=FULL_PRIMARY_DEFAULT)
    parser.add_argument("--ablation-root", type=Path, default=ABLATION_DEFAULT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_pooling(full_primary_dir=args.full_primary_dir, ablation_root=args.ablation_root, output_root=args.output_root)
    except Exception as exc:
        print(f"patient-level pooling refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
