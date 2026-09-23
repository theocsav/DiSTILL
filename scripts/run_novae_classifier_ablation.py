#!/usr/bin/env python3
"""Run the predeclared historical-164 NMF/NOVAE classifier ablation.

The validated ``run_novae_nmf_comparison.py`` output is an immutable reference.
Only the three non-full configurations are evaluated here; each configuration
is committed independently and the cross-configuration aggregate is published
last.  This module never edits or copies feature inputs or the full reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from scripts import run_novae_nmf_comparison as primary

EVALUATOR = primary.EVALUATOR
LAUNCHER = REPO / "scripts" / "submit_novae_classifier_ablation.sh"
ABLATIONS: dict[str, tuple[int, int]] = {
    "composition_only": (0, 0),
    "composition_enrichment": (5, 0),
    "composition_niche": (0, 20),
    "full": (5, 20),
}
NEW_ABLATIONS = tuple(name for name in ABLATIONS if name != "full")
CONFIGURATIONS = ABLATIONS
REQUIRED_ARTIFACTS = primary.REQUIRED_ARTIFACTS

class ContractError(ValueError):
    """Raised when a source, evaluator output, or manifest violates the contract."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _hash(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _table_path(stem: Path) -> Path:
    return primary._table_path(stem)


def _inventory(entries: list[tuple[str, Path]]) -> list[dict[str, str]]:
    return [{"name": name, "path": str(path), "sha256": _hash(path)} for name, path in entries]


def _primary_code_hashes() -> dict[str, str]:
    return {
        "evaluator": _hash(EVALUATOR),
        "orchestrator": _hash(REPO / "scripts" / "run_novae_nmf_comparison.py"),
        "launcher": _hash(REPO / "scripts" / "submit_novae_nmf_comparison.sh"),
    }


def protocol_for(ablation: str) -> dict[str, Any]:
    """Return the exact protocol declaration for one predeclared configuration."""
    if ablation not in ABLATIONS:
        raise ContractError(f"unknown ablation: {ablation}")
    top_enrichment, top_niche = ABLATIONS[ablation]
    protocol = dict(primary.PROTOCOL)
    protocol["top_enrichment"] = top_enrichment
    protocol["top_niche"] = top_niche
    if ablation != "full":
        protocol["ablation"] = ablation
        protocol["allowed_candidate_union"] = "composition + enrichment + niche_gene"
    return protocol


def expected_environment(ablation: str, prefix: str) -> dict[str, str]:
    top_enrichment, top_niche = ABLATIONS[ablation]
    values = dict(primary.EXPECTED)
    values.update(
        {
            "NICHERUNNER_TOP_ENRICHMENT_FEATURES": str(top_enrichment),
            "NICHERUNNER_TOP_NICHE_GENE_FEATURES": str(top_niche),
            "NICHERUNNER_COMPOSITION_PREFIX": prefix,
        }
    )
    return values


def _check_inherited_env(environment: dict[str, str] | None = None) -> None:
    source = os.environ if environment is None else environment
    forbidden = set(primary.SCIENTIFIC_ENV) | {"NICHERUNNER_MLP_OUTPUT_DIR", "NICHERUNNER_SOURCE_OUTPUT_DIR"}
    for name in forbidden:
        if source.get(name):
            raise ContractError(f"inherited {name} is forbidden; the ablation sets it explicitly")


def _arm_env(base: dict[str, str], feature_dir: Path, source_dir: Path, output_dir: Path, prefix: str, ablation: str) -> dict[str, str]:
    env = dict(base)
    env.update(expected_environment(ablation, prefix))
    env.update(
        {
            "NICHERUNNER_OUTPUT_DIR": str(feature_dir),
            "NICHERUNNER_SOURCE_OUTPUT_DIR": str(source_dir),
            "NICHERUNNER_MLP_OUTPUT_DIR": str(output_dir),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "42",
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
        }
    )
    return env


def _run_arms(envs: list[dict[str, str]], outputs: list[Path]) -> list[tuple[int, str]]:
    """Run NMF and NOVAE concurrently; each process is restricted to one thread."""
    processes = []
    stderr_paths = []
    try:
        for env, output in zip(envs, outputs, strict=True):
            stdout_path, stderr_path = output / "orchestrator_stdout.log", output / "orchestrator_stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen([sys.executable, str(EVALUATOR)], cwd=REPO, env=env, stdout=stdout_handle, stderr=stderr_handle, text=True)
            finally:
                stdout_handle.close()
                stderr_handle.close()
            processes.append(process)
            stderr_paths.append(stderr_path)
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    failed = False
    while True:
        statuses = [process.poll() for process in processes]
        if any(status not in (None, 0) for status in statuses):
            failed = True
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            break
        if all(status is not None for status in statuses):
            break
        time.sleep(0.2)
    results = []
    for process, stderr_path in zip(processes, stderr_paths, strict=True):
        if failed and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        results.append((process.returncode or 0, stderr_path.read_text(encoding="utf-8")))
    return results


def _validate_results_text(directory: Path, prefix: str, ablation: str) -> None:
    text = (directory / "mlp_results.txt").read_text(encoding="utf-8")
    required = [
        "Evaluation unit: fov", f"Composition prefix: {prefix}",
        "MLP backend: sklearn", "MLP device: cpu", "MLP max epochs: 1000", "MLP patience: 20",
        "MLP selection metric: weighted_f1", "MLP grid profile: compact", "MLP resampling: none",
        "MLP decision threshold: 0.5", "MLP mode: nested_cv", "Skip SHAP: True",
        "--- Final Performance Report ---",
    ]
    missing = [item for item in required if item not in text]
    if missing or any(f"--- Processing Outer Fold {fold}/14 ---" not in text for fold in range(1, 15)):
        raise ContractError(f"{directory.name} {ablation} output is missing exact protocol/fold metadata: {missing}")


def _require_exact_family_counts(features: list[str], enrichment: list[str], niche: list[str], top_enrichment: int, top_niche: int, label: str) -> None:
    enrichment_count = sum(feature in enrichment for feature in features)
    niche_count = sum(feature in niche for feature in features)
    if enrichment_count != top_enrichment or niche_count != top_niche:
        raise ContractError(f"{label} does not contain exactly its declared candidate family counts")


def _validate_paired_prediction_columns(nmf: pd.DataFrame, novae: pd.DataFrame, label: str) -> None:
    paired_columns = ("item_id", "true_label", "test_group", "outer_fold", "decision_threshold", "positive_class")
    for column in paired_columns:
        if not nmf[column].astype(str).equals(novae[column].astype(str)):
            raise ContractError(f"{label} paired predictions are not aligned in {column}")


def _validate_selected(directory: Path, arm: str, prefix: str, ablation: str, composition: list[str], enrichment: list[str], niche: list[str], predictions: pd.DataFrame, canonical: pd.Index, groups: pd.Series) -> None:
    top_enrichment, top_niche = ABLATIONS[ablation]
    allowed = set(composition) | set(enrichment) | set(niche)
    selected = pd.read_csv(directory / "selected_features_by_fold.csv")
    if len(selected) != 14 or selected.outer_fold.duplicated().any() or set(pd.to_numeric(selected.outer_fold, errors="raise").astype(int)) != set(range(1, 15)):
        raise ContractError(f"{arm} {ablation} selected-feature report must contain exactly one row for folds 1..14")
    prediction_groups = predictions.groupby("outer_fold", sort=True)["test_group"].first().to_dict()
    for _, row in selected.iterrows():
        fold = int(row.outer_fold)
        features = [feature for feature in str(row.get("selected_features", "")).split("|") if feature]
        if int(row.selected_feature_count) != len(features) or len(features) != len(set(features)):
            raise ContractError(f"{arm} {ablation} selected feature count/list is inconsistent")
        if not set(features).issubset(allowed):
            raise ContractError(f"{arm} {ablation} selected feature is outside the allowed candidate union")
        if not set(composition).issubset(features):
            raise ContractError(f"{arm} {ablation} selected composition columns are absent from fold {fold}")
        _require_exact_family_counts(features, enrichment, niche, top_enrichment, top_niche, f"{arm} {ablation} fold {fold}")
        train_groups = sorted(set(groups.astype(str).unique()) - {str(prediction_groups[fold])})
        if primary._group_text(row.test_groups) != str(prediction_groups[fold]) or primary._group_text(row.train_groups) != ",".join(train_groups):
            raise ContractError(f"{arm} {ablation} train/test group metadata disagrees for fold {fold}")
        fold_rows = predictions[predictions.outer_fold == fold]
        if int(row.test_rows) != len(fold_rows) or int(row.train_rows) != len(canonical) - len(fold_rows):
            raise ContractError(f"{arm} {ablation} train/test row metadata disagrees for fold {fold}")
    payload = json.loads((directory / "best_params.json").read_text(encoding="utf-8"))
    records = payload.get("outer_folds") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != 14:
        raise ContractError(f"{arm} {ablation} best_params.json must contain 14 folds")
    selected_by_fold = {int(row.outer_fold): row for _, row in selected.iterrows()}
    seen = set()
    for record in records:
        fold = int(record.get("outer_fold", -1))
        if fold in seen or fold not in range(1, 15):
            raise ContractError(f"{arm} {ablation} best_params.json has duplicate/invalid folds")
        seen.add(fold)
        row = selected_by_fold[fold]
        expected_features = [feature for feature in str(row.selected_features).split("|") if feature]
        expected_train = sorted(set(groups.astype(str).unique()) - {str(prediction_groups[fold])})
        if primary._group_text(record.get("test_groups")) != str(prediction_groups[fold]) or sorted(str(item) for item in record.get("train_groups", [])) != expected_train or int(record.get("test_rows", -1)) != int(row.test_rows) or int(record.get("train_rows", -1)) != int(row.train_rows) or int(record.get("selected_feature_count", -1)) != len(expected_features) or record.get("selected_features", []) != expected_features:
            raise ContractError(f"{arm} {ablation} best_params fold metadata disagrees")
        primary._validate_params(record.get("best_params", {}), f"{arm} {ablation} fold {fold}")
    fixed = json.loads((directory / "fixed_params.json").read_text(encoding="utf-8"))
    if fixed.get("selection_scope") != "grouped_full_data" or fixed.get("selection_metric") != "weighted_f1" or fixed.get("grid_profile") != "compact" or fixed.get("resampling") != "none" or fixed.get("backend") != "sklearn" or fixed.get("device") != "cpu" or fixed.get("max_epochs") != 1000 or fixed.get("patience") != 20:
        raise ContractError(f"{arm} {ablation} fixed_params.json protocol differs")
    primary._validate_params(fixed.get("best_params", {}), f"{arm} {ablation} fixed params")
    labels = sorted(predictions.true_label.astype(str).unique())
    stored = pd.read_csv(directory / "confusion_matrix.csv", index_col=0)
    expected = confusion_matrix(predictions.true_label.astype(str), predictions.predicted_label.astype(str), labels=labels)
    if stored.index.astype(str).tolist() != labels or stored.columns.astype(str).tolist() != labels or not np.array_equal(stored.to_numpy(dtype=float), expected.astype(float)):
        raise ContractError(f"{arm} {ablation} confusion matrix differs from predictions")


def validate_config_output(directory: Path, ablation: str, nmf: dict[str, Any], novae: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly validate one immutable configuration output and return predictions."""
    if ablation not in ABLATIONS:
        raise ContractError(f"unknown ablation: {ablation}")
    canonical, targets, groups = nmf["combined"].index, nmf["targets"], nmf["groups"]
    for name in REQUIRED_ARTIFACTS:
        for arm in ("nmf", "novae"):
            if not (directory / arm / name).is_file():
                raise ContractError(f"missing evaluator artifact: {directory / arm / name}")
    n = primary._predictions(directory / "nmf", canonical, targets, groups)
    v = primary._predictions(directory / "novae", canonical, targets, groups)
    _validate_results_text(directory / "nmf", "nmf_prop_", ablation)
    _validate_results_text(directory / "novae", "novae_prop_", ablation)
    args = [("nmf", "nmf_prop_", nmf), ("novae", "novae_prop_", novae)]
    for arm, prefix, source in args:
        _validate_selected(directory / arm, arm, prefix, ablation, source["composition_columns"], [str(c) for c in source["enrichment"].columns], [str(c) for c in source["niche"].columns], n if arm == "nmf" else v, canonical, groups)
    _validate_paired_prediction_columns(n, v, ablation)
    return n, v


def _feature_reports(directory: Path, arm: str, composition: list[str], ablation: str) -> pd.DataFrame:
    selected = pd.read_csv(directory / "selected_features_by_fold.csv")
    counts: dict[str, int] = {}
    for _, row in selected.iterrows():
        features = [feature for feature in str(row.selected_features).split("|") if feature]
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
    rows = [{"ablation": ablation, "arm": arm, "feature": feature, "fold_count": count, "fold_frequency": count / 14} for feature, count in sorted(counts.items())]
    return pd.DataFrame(rows)


def summarize_config(directory: Path, ablation: str, nmf: dict[str, Any], novae: dict[str, Any], input_hashes: dict[str, Any]) -> dict[str, Any]:
    n, v = validate_config_output(directory, ablation, nmf, novae)
    labels = sorted(nmf["targets"].unique().tolist())
    nmf_metrics = primary._metrics(n.true_label, n.predicted_label, labels)
    novae_metrics = primary._metrics(v.true_label, v.predicted_label, labels)
    joined = pd.DataFrame({"item_id": n.item_id.astype(str), "test_group": n.test_group.astype(str), "true_label": n.true_label.astype(str), "nmf_predicted_label": n.predicted_label.astype(str), "novae_predicted_label": v.predicted_label.astype(str)})
    joined["nmf_correct"] = joined.nmf_predicted_label == joined.true_label
    joined["novae_correct"] = joined.novae_predicted_label == joined.true_label
    joined.to_csv(directory / "paired_predictions.csv", index=False)
    patient_rows = []
    for patient, rows in joined.groupby("test_group", sort=True):
        n_correct, v_correct = int(rows.nmf_correct.sum()), int(rows.novae_correct.sum())
        patient_rows.append({"ablation": ablation, "patient": patient, "row_count": len(rows), "nmf_accuracy": n_correct / len(rows), "novae_accuracy": v_correct / len(rows), "delta_accuracy_novae_minus_nmf": (v_correct - n_correct) / len(rows)})
    pd.DataFrame(patient_rows).to_csv(directory / "per_patient_metrics.csv", index=False)
    pd.DataFrame(nmf_metrics["confusion_matrix"], index=labels, columns=labels).to_csv(directory / "confusion_matrix_nmf.csv")
    pd.DataFrame(novae_metrics["confusion_matrix"], index=labels, columns=labels).to_csv(directory / "confusion_matrix_novae.csv")
    feature_frames = []
    for arm, source in (("nmf", nmf), ("novae", novae)):
        report = _feature_reports(directory / arm, arm, source["composition_columns"], ablation)
        report.to_csv(directory / f"selected_feature_frequencies_{arm}.csv", index=False)
        feature_frames.append(report)
    feature_stability = pd.concat(feature_frames, ignore_index=True)
    feature_stability.to_csv(directory / "feature_stability.csv", index=False)
    delta = {metric: novae_metrics[metric] - nmf_metrics[metric] for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
    rows = [{"ablation": ablation, "representation": representation, "metric": metric, "value": metrics[metric], "delta_novae_minus_nmf": delta[metric] if representation == "novae" else 0.0} for representation, metrics in (("nmf", nmf_metrics), ("novae", novae_metrics)) for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")]
    pd.DataFrame(rows).to_csv(directory / "pooled_metrics_long.csv", index=False)
    summary = {"ablation": ablation, "protocol": protocol_for(ablation), "arms": {"nmf": nmf_metrics, "novae": novae_metrics}, "delta_novae_minus_nmf": delta, "inputs": input_hashes, "warning": "NOVAE remains exploratory reference=all; no FOV independence or p-value claim is made."}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _expected_input_hashes(nmf_feature_dir: Path, nmf_source_dir: Path, novae_feature_dir: Path, novae_manifest: dict[str, Any]) -> dict[str, Any]:
    nmf_enrichment, nmf_niche = _table_path(nmf_source_dir / "enrichment_features_fov"), _table_path(nmf_source_dir / "niche_gene_features_fov")
    novae_enrichment, novae_niche = _table_path(novae_feature_dir / "enrichment_features_fov"), _table_path(novae_feature_dir / "niche_gene_features_fov")
    return {
        "evaluator": _hash(EVALUATOR),
        "nmf": _inventory([("combined_features_filtered.parquet", nmf_feature_dir / "combined_features_filtered.parquet"), ("targets_y.parquet", nmf_feature_dir / "targets_y.parquet"), ("groups.parquet", nmf_feature_dir / "groups.parquet"), ("enrichment_features_fov", nmf_enrichment), ("niche_gene_features_fov", nmf_niche), ("post_nmf_obs.csv", nmf_source_dir / "post_nmf_obs.csv")]),
        "novae": _inventory([("combined_features_filtered.parquet", novae_feature_dir / "combined_features_filtered.parquet"), ("targets_y.parquet", novae_feature_dir / "targets_y.parquet"), ("groups.parquet", novae_feature_dir / "groups.parquet"), ("enrichment_features_fov", novae_enrichment), ("niche_gene_features_fov", novae_niche), ("novae_feature_manifest.json", novae_feature_dir / "novae_feature_manifest.json")]),
        "novae_contract": {"contract": novae_manifest.get("contract"), "warning": novae_manifest.get("warning"), "provenance": novae_manifest.get("novae_pilot_provenance")},
    }


def validate_full_primary(full_dir: Path, nmf: dict[str, Any], novae: dict[str, Any], expected_inputs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate job 43018668's output without writing to it."""
    full_dir = Path(full_dir)
    manifest_path = full_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ContractError(f"missing full primary run_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != primary.PROTOCOL:
        raise ContractError("full primary protocol does not exactly match the frozen primary protocol")
    expected_code = _primary_code_hashes()
    if manifest.get("code_sha256") != expected_code:
        raise ContractError("full primary code hashes differ from current immutable primary files")
    if manifest.get("input_sha256") != expected_inputs:
        raise ContractError("full primary input hashes/paths do not exactly match declared inputs")
    inventory = manifest.get("output_sha256")
    actual_inventory = {name: digest for name, digest in primary._tree_hashes(full_dir).items() if name != "run_manifest.json"}
    if not isinstance(inventory, dict) or inventory != actual_inventory:
        raise ContractError("full primary output hash inventory is missing or does not match")
    # The primary validator has the full (5,20) limits and validates the same
    # artifact set; call it directly so the source output is only read.
    canonical, targets, groups = nmf["combined"].index, nmf["targets"], nmf["groups"]
    n = primary._predictions(full_dir / "nmf", canonical, targets, groups)
    v = primary._predictions(full_dir / "novae", canonical, targets, groups)
    # Validate both the frozen primary artifact contract and this ablation's
    # dynamic exact family-count contract without writing to the reference.
    primary._validate_arm_artifacts(full_dir / "nmf", "nmf", "nmf_prop_", nmf["composition_columns"], [str(c) for c in nmf["enrichment"].columns], [str(c) for c in nmf["niche"].columns], n, canonical, groups)
    primary._validate_arm_artifacts(full_dir / "novae", "novae", "novae_prop_", novae["composition_columns"], [str(c) for c in novae["enrichment"].columns], [str(c) for c in novae["niche"].columns], v, canonical, groups)
    _validate_selected(full_dir / "nmf", "nmf", "nmf_prop_", "full", nmf["composition_columns"], [str(c) for c in nmf["enrichment"].columns], [str(c) for c in nmf["niche"].columns], n, canonical, groups)
    _validate_selected(full_dir / "novae", "novae", "novae_prop_", "full", novae["composition_columns"], [str(c) for c in novae["enrichment"].columns], [str(c) for c in novae["niche"].columns], v, canonical, groups)
    _validate_paired_prediction_columns(n, v, "full primary")
    return n, v, manifest


def _metrics_long(ablation: str, n: pd.DataFrame, v: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    nmf_metrics, novae_metrics = primary._metrics(n.true_label, n.predicted_label, labels), primary._metrics(v.true_label, v.predicted_label, labels)
    rows = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        rows.extend([{"ablation": ablation, "representation": "nmf", "metric": metric, "value": nmf_metrics[metric], "delta_novae_minus_nmf": novae_metrics[metric] - nmf_metrics[metric]}, {"ablation": ablation, "representation": "novae", "metric": metric, "value": novae_metrics[metric], "delta_novae_minus_nmf": novae_metrics[metric] - nmf_metrics[metric]}])
    return pd.DataFrame(rows)


def _paired_long(ablation: str, n: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"ablation": ablation, "item_id": n.item_id.astype(str), "patient": n.test_group.astype(str), "true_label": n.true_label.astype(str), "nmf_predicted_label": n.predicted_label.astype(str), "novae_predicted_label": v.predicted_label.astype(str)})
    result["nmf_correct"] = result.nmf_predicted_label == result.true_label
    result["novae_correct"] = result.novae_predicted_label == result.true_label
    return result


def _patient_long(ablation: str, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, group in paired.groupby("patient", sort=True):
        n_correct, v_correct = int(group.nmf_correct.sum()), int(group.novae_correct.sum())
        rows.append({"ablation": ablation, "patient": patient, "row_count": len(group), "nmf_accuracy": n_correct / len(group), "novae_accuracy": v_correct / len(group), "delta_accuracy_novae_minus_nmf": (v_correct - n_correct) / len(group)})
    return pd.DataFrame(rows)


def _load_config_summary(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.DataFrame([{ "ablation": summary["ablation"], "representation": rep, "metric": metric, "value": summary["arms"][rep][metric], "delta_novae_minus_nmf": summary["delta_novae_minus_nmf"][metric]} for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1") for rep in ("nmf", "novae")])
    paired = pd.read_csv(directory / "paired_predictions.csv")
    paired.insert(0, "ablation", summary["ablation"])
    patients = pd.read_csv(directory / "per_patient_metrics.csv")
    return metrics, paired, patients


def publish_aggregate(root: Path, full_dir: Path, full_n: pd.DataFrame, full_v: pd.DataFrame, completed: list[str], nmf: dict[str, Any], novae: dict[str, Any], expected_inputs: dict[str, Any]) -> dict[str, Any]:
    target = root / "aggregate"
    if target.exists():
        raise ContractError(f"refusing existing aggregate: {target}")
    stage = Path(tempfile.mkdtemp(prefix=".aggregate.", dir=root))
    try:
        metric_frames, paired_frames, patient_frames, stability_frames = [], [], [], []
        labels = sorted(nmf["targets"].unique().tolist())
        metric_frames.append(_metrics_long("full", full_n, full_v, labels))
        paired_frames.append(_paired_long("full", full_n, full_v))
        patient_frames.append(_patient_long("full", paired_frames[-1]))
        full_stability = []
        for arm in ("nmf", "novae"):
            frame = _feature_reports(full_dir / arm, arm, nmf["composition_columns"] if arm == "nmf" else novae["composition_columns"], "full")
            frame.to_csv(stage / f"selected_feature_frequencies_{arm}_full.csv", index=False)
            full_stability.append(frame)
        stability_frames.extend(full_stability)
        for ablation in completed:
            metrics, paired, patients = _load_config_summary(root / ablation)
            metric_frames.append(metrics)
            paired_frames.append(paired)
            patient_frames.append(patients)
            stability_frames.append(pd.read_csv(root / ablation / "feature_stability.csv"))
        pd.concat(metric_frames, ignore_index=True).to_csv(stage / "pooled_metrics_long.csv", index=False)
        pd.concat(paired_frames, ignore_index=True).to_csv(stage / "paired_predictions_long.csv", index=False)
        pd.concat(patient_frames, ignore_index=True).to_csv(stage / "per_patient_metrics_long.csv", index=False)
        pd.concat(stability_frames, ignore_index=True).to_csv(stage / "feature_stability_long.csv", index=False)
        matrix = pd.concat(metric_frames, ignore_index=True)
        matrix.to_csv(stage / "ablation_metric_matrix.csv", index=False)
        aggregate = {"protocol": {name: protocol_for(name) for name in ABLATIONS}, "completed_ablations": ["full", *completed], "full_primary_output": str(full_dir), "full_primary_manifest_sha256": _hash(full_dir / "run_manifest.json"), "full_primary_unchanged": True, "inputs": expected_inputs, "code_sha256": {"evaluator": _hash(EVALUATOR), "orchestrator": _hash(Path(__file__)), "launcher": _hash(LAUNCHER)}, "warning": "NOVAE remains exploratory reference=all; no FOV independence or p-value claim is made."}
        aggregate["output_sha256"] = _tree_hashes(stage)
        aggregate["output_sha256_excludes"] = ["run_manifest.json"]
        (stage / "run_manifest.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        os.rename(stage, target)
        return aggregate
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run_ablation(*, nmf_feature_dir: Path, nmf_source_dir: Path, novae_feature_dir: Path, full_primary_dir: Path, output_root: Path, base_env: dict[str, str] | None = None, candidate_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Run/resume the three new arms and atomically publish their aggregate."""
    output_root = Path(output_root)
    _check_inherited_env(base_env if base_env is not None else None)
    if output_root.exists() and not output_root.is_dir():
        raise ContractError(f"output root is not a directory: {output_root}")
    output_root = output_root.resolve(strict=False)
    source_paths = [Path(nmf_feature_dir), Path(nmf_source_dir), Path(novae_feature_dir), Path(full_primary_dir)]
    for source in source_paths:
        source = source.resolve(strict=False)
        if output_root == source or output_root in source.parents or source in output_root.parents:
            raise ContractError(f"output root collides with input path: {output_root} / {source}")
    output_root.mkdir(parents=True, exist_ok=True)
    unknown = set(output_root.iterdir()) - {output_root / name for name in (*NEW_ABLATIONS, "aggregate")}
    if unknown:
        raise ContractError(f"output root contains unexpected entries: {sorted(str(path) for path in unknown)}")
    if (output_root / "aggregate").exists():
        raise ContractError("aggregate already exists; refusing to alter a published result")
    nmf = primary.preflight_arm(Path(nmf_feature_dir), Path(nmf_source_dir), prefix="nmf_prop_", candidate_counts=candidate_counts)
    nmf["composition_columns"] = [str(c) for c in nmf["composition_columns"]]
    novae_manifest = primary.validate_novae_manifest(Path(novae_feature_dir))
    novae = primary.preflight_arm(Path(novae_feature_dir), Path(novae_feature_dir), prefix="novae_prop_", canonical=nmf["combined"].index, expected_targets=nmf["targets"], expected_groups=nmf["groups"], candidate_counts=candidate_counts)
    expected_inputs = _expected_input_hashes(Path(nmf_feature_dir), Path(nmf_source_dir), Path(novae_feature_dir), novae_manifest)
    full_n, full_v, _ = validate_full_primary(Path(full_primary_dir), nmf, novae, expected_inputs)
    completed = []
    envbase = dict(base_env or os.environ)
    for ablation in NEW_ABLATIONS:
        target = output_root / ablation
        if target.exists():
            if not target.is_dir() or not (target / "config_manifest.json").is_file():
                raise ContractError(f"existing ablation output is not a complete immutable configuration: {target}")
            stored = json.loads((target / "config_manifest.json").read_text(encoding="utf-8"))
            expected_protocol = protocol_for(ablation)
            expected_env = {"nmf": expected_environment(ablation, "nmf_prop_"), "novae": expected_environment(ablation, "novae_prop_")}
            actual_inventory = {name: digest for name, digest in _tree_hashes(target).items() if name != "config_manifest.json"}
            if stored.get("protocol") != expected_protocol or stored.get("inputs") != expected_inputs or stored.get("environment") != expected_env or stored.get("output_sha256") != actual_inventory:
                raise ContractError(f"existing {ablation} manifest does not match exact protocol/inputs/environment/output hashes")
            if stored.get("code_sha256", {}).get("evaluator") != _hash(EVALUATOR) or stored.get("code_sha256", {}).get("orchestrator") != _hash(Path(__file__)) or stored.get("code_sha256", {}).get("launcher") != _hash(LAUNCHER):
                raise ContractError(f"existing {ablation} manifest code hash mismatch")
            validate_config_output(target, ablation, nmf, novae)
            completed.append(ablation)
            continue
        stage = Path(tempfile.mkdtemp(prefix=f".{ablation}.", dir=output_root))
        try:
            (stage / "nmf").mkdir(); (stage / "novae").mkdir()
            envs = [_arm_env(envbase, Path(nmf_feature_dir), Path(nmf_source_dir), stage / "nmf", "nmf_prop_", ablation), _arm_env(envbase, Path(novae_feature_dir), Path(novae_feature_dir), stage / "novae", "novae_prop_", ablation)]
            statuses = _run_arms(envs, [stage / "nmf", stage / "novae"])
            if any(code != 0 for code, _ in statuses):
                raise ContractError(f"{ablation} evaluator arms failed")
            summarize_config(stage, ablation, nmf, novae, expected_inputs)
            config_manifest = {"ablation": ablation, "protocol": protocol_for(ablation), "inputs": expected_inputs, "environment": {"nmf": expected_environment(ablation, "nmf_prop_"), "novae": expected_environment(ablation, "novae_prop_")}, "code_sha256": {"evaluator": _hash(EVALUATOR), "orchestrator": _hash(Path(__file__)), "launcher": _hash(LAUNCHER)}}
            config_manifest["output_sha256"] = _tree_hashes(stage)
            config_manifest["output_sha256_excludes"] = ["config_manifest.json"]
            (stage / "config_manifest.json").write_text(json.dumps(config_manifest, indent=2) + "\n", encoding="utf-8")
            if target.exists():
                raise ContractError(f"refusing existing ablation output: {target}")
            os.rename(stage, target)
            completed.append(ablation)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return publish_aggregate(output_root, Path(full_primary_dir), full_n, full_v, completed, nmf, novae, expected_inputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmf-feature-dir", type=Path, required=True)
    parser.add_argument("--nmf-source-dir", type=Path, required=True)
    parser.add_argument("--novae-feature-dir", type=Path, required=True)
    parser.add_argument("--full-primary-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_ablation(nmf_feature_dir=args.nmf_feature_dir, nmf_source_dir=args.nmf_source_dir, novae_feature_dir=args.novae_feature_dir, full_primary_dir=args.full_primary_dir, output_root=args.output_root)
    except Exception as exc:
        print(f"classifier ablation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
