#!/usr/bin/env python3
"""Run the frozen historical-164 NMF-only versus NOVAE-only comparison.

This is an orchestration and audit layer around ``IBD_MLP_LeakageSafe.py``.  It
never changes an input and publishes a result only after both arms and all
alignment checks pass.
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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[1]
EVALUATOR = REPO / "pipeline_assets" / "IBD_MLP_LeakageSafe.py"
NMF_FEATURE_DIR = Path("/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_inputs")
NMF_SOURCE_DIR = Path("/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs")
NOVAE_FEATURE_DIR = Path("/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_res1_fov_features_historical164_typed_v2/features")
DEFAULT_OUTPUT_ROOT = REPO / "runs" / "novae_nmf_comparison"

EXPECTED = {
    "NICHERUNNER_MLP_UNIT": "fov", "NICHERUNNER_MLP_MODE": "nested_cv",
    "NICHERUNNER_MLP_BACKEND": "sklearn", "NICHERUNNER_MLP_GRID_PROFILE": "compact",
    "NICHERUNNER_MLP_SELECTION_METRIC": "weighted_f1", "NICHERUNNER_MLP_RESAMPLING": "none",
    "NICHERUNNER_MLP_DECISION_THRESHOLD": "0.5", "NICHERUNNER_MLP_MAX_EPOCHS": "1000",
    "NICHERUNNER_MLP_PATIENCE": "20", "NICHERUNNER_TOP_ENRICHMENT_FEATURES": "5",
    "NICHERUNNER_TOP_NICHE_GENE_FEATURES": "20", "NICHERUNNER_SKIP_SHAP": "1",
    "NICHERUNNER_MLP_DEVICE": "cpu",
}
SCIENTIFIC_ENV = set(EXPECTED) | {"NICHERUNNER_COMPOSITION_PREFIX", "NICHERUNNER_MLP_FIXED_PARAMS_PATH", "NICHERUNNER_MLP_BEST_PARAMS_OUT"}
REQUIRED_ARTIFACTS = ("mlp_results.txt", "confusion_matrix.csv", "best_params.json", "selected_features_by_fold.csv", "fold_predictions.csv", "fixed_params.json")
EXPECTED_COMPOSITION_COUNT = 9
EXPECTED_CANDIDATE_COUNTS = {"enrichment": 81, "niche": {"nmf_prop_": 162343, "novae_prop_": 162666}}


class ContractError(ValueError):
    pass


def _read_table(stem: Path) -> pd.DataFrame:
    path = stem if stem.suffix else (stem.with_suffix(".parquet") if stem.with_suffix(".parquet").exists() else stem.with_suffix(".csv"))
    if not path.exists():
        raise ContractError(f"missing table: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ContractError(f"duplicate index in {path}")
    if frame.columns.duplicated().any():
        raise ContractError(f"duplicate columns in {path}")
    return frame


def _series(path: Path) -> pd.Series:
    frame = _read_table(path)
    if frame.shape[1] != 1:
        raise ContractError(f"{path} must contain one column")
    return frame.iloc[:, 0]


def _finite(frame: pd.DataFrame, name: str) -> np.ndarray:
    try:
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} is not numeric") from exc
    if not np.isfinite(values).all():
        raise ContractError(f"{name} contains non-finite values")
    return values


def _validate_composition(frame: pd.DataFrame, columns: list[str], name: str, *, allow_zero: bool) -> None:
    values = _finite(frame.loc[:, columns], name)
    if (values < 0).any():
        raise ContractError(f"{name} contains negative proportions")
    sums = values.sum(axis=1)
    if allow_zero:
        valid = np.isclose(sums, 1.0, rtol=1e-6, atol=1e-8) | np.isclose(sums, 0.0, atol=1e-8)
    else:
        valid = np.isclose(sums, 1.0, rtol=1e-6, atol=1e-8)
    if not valid.all():
        raise ContractError(f"{name} rows are not normalized composition proportions")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _hash(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _table_path(stem: Path) -> Path:
    if stem.suffix:
        path = stem
    elif stem.with_suffix(".parquet").exists():
        path = stem.with_suffix(".parquet")
    else:
        path = stem.with_suffix(".csv")
    if not path.exists():
        raise ContractError(f"missing table: {path}")
    return path


def _input_inventory(entries: list[tuple[str, Path]]) -> list[dict[str, str]]:
    return [{"name": label, "path": str(path), "sha256": _hash(path)} for label, path in entries]


def _pick(columns: Any, names: tuple[str, ...], label: str) -> str:
    for name in names:
        if name in columns:
            return name
    raise ContractError(f"missing {label}; expected one of {names}")


def _authoritative_mapping(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "post_nmf_obs.csv"
    if not path.exists():
        raise ContractError(f"missing authoritative mapping: {path}")
    frame = pd.read_csv(path, index_col=0)
    fov_col = _pick(frame.columns, ("field_of_view", "unique_fov", "fov_key", "fov"), "field_of_view")
    patient_col = _pick(frame.columns, ("patient", "Patient", "subject", "sample_id"), "patient")
    disease_col = _pick(frame.columns, ("Disease_State", "disease_state", "Disease/Health State", "Disease.Health.State"), "disease")
    fov = frame[fov_col].astype(str)
    if fov_col == "fov" and "patient" in frame:
        fov = frame[patient_col].astype(str) + "_" + fov
    result = pd.DataFrame({"fov": fov.to_numpy(), "patient": frame[patient_col].astype(str).to_numpy(), "label": frame[disease_col].astype(str).to_numpy()}, index=frame.index.astype(str))
    if result.isna().any().any() or result.astype(str).apply(lambda column: column.str.strip().isin({"", "nan", "none", "null"})).any().any():
        raise ContractError("authoritative post_nmf_obs contains blank metadata")
    bad = result.groupby("fov", sort=False).agg({"patient": "nunique", "label": "nunique"})
    if (bad > 1).any().any():
        raise ContractError("authoritative mapping has mixed patient/disease metadata within an FOV")
    return result


def _expected_nmf_columns(mapping: pd.DataFrame) -> list[str]:
    # The NMF factor is the authoritative composition source, never a selected
    # feature table.  This function is passed a frame carrying it by preflight.
    if "nmf_factor" not in mapping:
        raise ContractError("authoritative post_nmf_obs is missing NMF_factor")
    factors = mapping["nmf_factor"].astype(str)
    if factors.str.strip().isin({"", "nan", "none"}).any():
        raise ContractError("authoritative NMF_factor contains missing values")
    labels = sorted(factors.unique(), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    return [f"nmf_prop_{value}" for value in labels]


def validate_novae_manifest(feature_dir: Path) -> dict[str, Any]:
    path = feature_dir / "novae_feature_manifest.json"
    if not path.exists():
        raise ContractError(f"missing NOVAE manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "historical_164":
        raise ContractError("NOVAE manifest is not historical_164")
    warning = str(manifest.get("warning", ""))
    if "reference=all" not in warning or "exploratory" not in warning:
        raise ContractError("NOVAE manifest must retain the exploratory reference=all warning")
    provenance = manifest.get("novae_pilot_provenance")
    if not isinstance(provenance, dict):
        raise ContractError("NOVAE manifest lacks typed provenance")
    expected = {"analysis_scope": "exploratory", "reference": "all", "inference_mode": "zero_shot", "dataset_id": "skin_visium_ssc_paired_cpu_calibrated", "coordinate_strategy": "visium_explicit_scale", "primary_resolution": 1.0, "domain_key": "novae_domains_res1.0", "neighborhood_valid_key": "neighborhood_valid", "accelerator": "cpu", "device": "cpu", "workers": 0, "seed": 42, "confirmatory_held_out_classification_allowed": False, "minimum_domain_assignment_coverage": 0.70, "input_sha256": "262418e8e7ed06de805e940406f3ae9e41487ce085da1ae8f940c81f95daf6dd", "checkpoint_sha256": "1422f9f72d6e532921bf8a90f0996f1c46c6891f6ecbc73e404521ec5aa7b04a"}
    for key, value in expected.items():
        if key not in provenance or provenance[key] != value or isinstance(provenance[key], str) != isinstance(value, str):
            raise ContractError(f"NOVAE provenance mismatch or non-native type for {key}")
    policy = provenance.get("deterministic_policy")
    if not isinstance(policy, dict) or policy.get("requested") is not True or policy.get("effective") is not True or isinstance(policy.get("requested"), str) or isinstance(policy.get("effective"), str):
        raise ContractError("NOVAE deterministic provenance is not native typed true/true")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ContractError("NOVAE manifest has no output hash inventory")
    for relative, entry in outputs.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ContractError(f"invalid NOVAE output hash entry: {relative}")
        output = feature_dir / relative
        if not output.is_file() or _hash(output) != entry["sha256"]:
            raise ContractError(f"NOVAE output hash mismatch: {relative}")
    return manifest


def preflight_arm(feature_dir: Path, source_dir: Path | None, *, prefix: str, canonical: pd.Index | None = None, expected_targets: pd.Series | None = None, expected_groups: pd.Series | None = None, candidate_counts: dict[str, int] | None = None) -> dict[str, Any]:
    combined = _read_table(feature_dir / "combined_features_filtered.parquet")
    targets = _series(feature_dir / "targets_y.parquet").astype(str)
    groups = _series(feature_dir / "groups.parquet").astype(str)
    if len(combined) != 164 or not combined.index.is_unique:
        raise ContractError(f"{prefix} arm must have exactly 164 unique FOV rows")
    if canonical is not None and not combined.index.equals(canonical):
        raise ContractError(f"{prefix} combined index/order differs from NMF canonical index")
    if not targets.index.equals(combined.index) or not groups.index.equals(combined.index):
        raise ContractError(f"{prefix} target/group index/order differs from combined")
    if targets.isna().any() or groups.isna().any():
        raise ContractError(f"{prefix} target/group contains missing values")
    if sorted(targets.value_counts().tolist()) != [61, 103] or groups.nunique() != 14:
        raise ContractError(f"{prefix} does not have frozen [61,103] / 14-group contract")
    if expected_targets is not None and not targets.equals(expected_targets):
        raise ContractError("target values/order differ between arms")
    if expected_groups is not None and not groups.equals(expected_groups):
        raise ContractError("group values/order differ between arms")
    composition = [str(c) for c in combined.columns if str(c).startswith(prefix)]
    forbidden = "novae_prop_" if prefix == "nmf_prop_" else "nmf_prop_"
    if any(str(c).startswith(forbidden) for c in combined.columns) or not composition:
        raise ContractError(f"{prefix} arm has forbidden/absent composition aliases")
    if prefix == "novae_prop_" and (len(composition) != len(combined.columns) or composition != [f"novae_prop_L{i}" for i in range(9)]):
        raise ContractError("NOVAE combined table must contain exactly novae_prop_L0..novae_prop_L8")
    if len(composition) != EXPECTED_COMPOSITION_COUNT:
        raise ContractError(f"{prefix} arm must have exactly {EXPECTED_COMPOSITION_COUNT} composition columns")
    _validate_composition(combined, composition, f"{prefix} composition", allow_zero=prefix == "novae_prop_")
    candidate_counts = candidate_counts or {"enrichment": EXPECTED_CANDIDATE_COUNTS["enrichment"], "niche": EXPECTED_CANDIDATE_COUNTS["niche"][prefix]}
    enrichment = _read_table((source_dir or feature_dir) / "enrichment_features_fov")
    niche = _read_table((source_dir or feature_dir) / "niche_gene_features_fov")
    if enrichment.shape[1] != candidate_counts["enrichment"] or niche.shape[1] != candidate_counts["niche"]:
        raise ContractError(f"{prefix} candidate feature counts differ from frozen contract")
    _finite(enrichment, f"{prefix} enrichment candidates")
    _finite(niche, f"{prefix} niche candidates")
    if not set(enrichment.index).issubset(set(combined.index)) or not set(niche.index).issubset(set(combined.index)):
        raise ContractError(f"{prefix} candidate table has extra FOV rows")
    if prefix == "nmf_prop_":
        mapping = _authoritative_mapping(source_dir or feature_dir)
        raw = pd.read_csv((source_dir or feature_dir) / "post_nmf_obs.csv", index_col=0)
        factor_col = _pick(raw.columns, ("NMF_factor", "nmf_factor", "dominant_nmf_factor"), "NMF_factor")
        mapping["nmf_factor"] = raw[factor_col].astype(str).to_numpy()
        if set(mapping.fov) != set(combined.index):
            raise ContractError("authoritative field_of_view mapping differs from canonical index")
        mapped = mapping.groupby("fov", sort=False).first().reindex(combined.index)
        if not mapped["patient"].astype(str).reset_index(drop=True).equals(groups.astype(str).reset_index(drop=True)) or not mapped["label"].astype(str).reset_index(drop=True).equals(targets.astype(str).reset_index(drop=True)):
            raise ContractError("authoritative field_of_view mapping disagrees with frozen targets/groups")
        expected_composition = _expected_nmf_columns(mapping)
        if composition != expected_composition:
            raise ContractError(f"NMF composition columns/order differ from expected factors: {expected_composition}")
        singleton = set(mapping.groupby("fov").size().loc[lambda s: s == 1].index)
        missing = set(combined.index) - set(enrichment.index)
        if not missing.issubset(singleton):
            raise ContractError("NMF enrichment is missing a non-singleton FOV")
        if not niche.index.equals(combined.index):
            raise ContractError("NMF niche candidate rows must exactly match canonical index/order")
    elif not enrichment.index.equals(combined.index) or not niche.index.equals(combined.index):
        raise ContractError("NOVAE candidate rows must exactly match canonical index/order")
    return {"combined": combined, "targets": targets, "groups": groups, "composition_columns": composition, "enrichment": enrichment, "niche": niche}


def _check_inherited_env(environment: dict[str, str] | None = None) -> None:
    source = os.environ if environment is None else environment
    for name in SCIENTIFIC_ENV:
        if source.get(name):
            raise ContractError(f"inherited {name} is forbidden; the comparison sets it explicitly")


def _arm_env(base: dict[str, str], feature_dir: Path, source_dir: Path, output_dir: Path, prefix: str) -> dict[str, str]:
    env = dict(base)
    env.update(EXPECTED)
    env.update({"NICHERUNNER_OUTPUT_DIR": str(feature_dir), "NICHERUNNER_SOURCE_OUTPUT_DIR": str(source_dir), "NICHERUNNER_MLP_OUTPUT_DIR": str(output_dir), "NICHERUNNER_COMPOSITION_PREFIX": prefix})
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "42", "CUDA_VISIBLE_DEVICES": ""})
    return env


def _run_arms(envs: list[dict[str, str]], outputs: list[Path]) -> list[tuple[int, str]]:
    processes = []
    stderr_paths = []
    try:
        for env, output in zip(envs, outputs):
            stdout_handle = (output / "orchestrator_stdout.log").open("w", encoding="utf-8")
            stderr_path = output / "orchestrator_stderr.log"
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
    for process in processes:
        if failed and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        err = stderr_paths[len(results)].read_text(encoding="utf-8")
        results.append((process.returncode or 0, err))
    return results


def _require_artifacts(directory: Path) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (directory / name).is_file():
            raise ContractError(f"evaluator missing artifact {directory / name}")


def _predictions(directory: Path, canonical: pd.Index, targets: pd.Series, groups: pd.Series) -> pd.DataFrame:
    _require_artifacts(directory)
    frame = pd.read_csv(directory / "fold_predictions.csv")
    required = {"item_id", "outer_fold", "test_group", "true_label", "predicted_label", "decision_threshold", "positive_class", "positive_class_probability"}
    if not required.issubset(frame.columns):
        raise ContractError(f"fold_predictions.csv missing columns: {required - set(frame.columns)}")
    if frame.item_id.astype(str).tolist() != canonical.astype(str).tolist() or frame.item_id.astype(str).duplicated().any():
        raise ContractError("prediction item_id order/set does not match canonical index")
    frame["item_id"] = frame.item_id.astype(str)
    frame["outer_fold"] = pd.to_numeric(frame.outer_fold, errors="raise").astype(int)
    if set(frame.outer_fold) != set(range(1, 15)):
        raise ContractError("predictions do not contain all 14 outer folds")
    fold_groups = frame.groupby("outer_fold", sort=True)["test_group"].agg(lambda values: set(values.astype(str)))
    if len(fold_groups) != 14 or not all(len(groups_for_fold) == 1 for groups_for_fold in fold_groups):
        raise ContractError("each outer fold must map one-to-one to one patient group")
    fold_to_group = {int(fold): next(iter(groups_for_fold)) for fold, groups_for_fold in fold_groups.items()}
    if set(fold_to_group.values()) != set(groups.astype(str).unique()):
        raise ContractError("outer folds do not cover exactly the 14 patient groups")
    if frame.predicted_label.isna().any() or frame.positive_class_probability.isna().any():
        raise ContractError("stored prediction fields contain missing values")
    probabilities = pd.to_numeric(frame.positive_class_probability, errors="coerce")
    if not np.isfinite(probabilities).all() or not probabilities.between(0.0, 1.0).all():
        raise ContractError("stored positive class probabilities are not finite values in [0,1]")
    if not frame.true_label.astype(str).equals(targets.reindex(canonical).reset_index(drop=True)):
        raise ContractError("stored true labels differ from frozen targets")
    expected_group = groups.reindex(canonical).reset_index(drop=True).astype(str)
    if not frame.test_group.astype(str).equals(expected_group):
        raise ContractError("stored test groups differ from frozen groups")
    if not np.allclose(pd.to_numeric(frame.decision_threshold), 0.5):
        raise ContractError("stored decision thresholds differ from .5")
    if frame.positive_class.astype(str).nunique() != 1 or frame.positive_class.astype(str).iloc[0] != "systemic_sclerosis":
        raise ContractError("positive class is not canonical across folds")
    return frame


def _metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), "confusion_matrix": cm.tolist()}


COMPACT_GRID = {
    "hidden_layer_sizes": {(16,), (32,), (32, 16), (32, 16, 8)},
    "activation": {"relu", "tanh"},
    "alpha": {1e-3, 1e-2},
    "learning_rate_init": {1e-3, 1e-2},
    "batch_size": {8, 16},
}


def _validate_params(params: dict[str, Any], label: str) -> None:
    required = set(COMPACT_GRID) | {"backend", "device", "max_epochs", "patience"}
    if set(params) != required:
        raise ContractError(f"{label} parameters differ from serialized protocol keys")
    try:
        hidden = tuple(int(value) for value in params["hidden_layer_sizes"])
        activation = str(params["activation"])
        alpha = float(params["alpha"])
        learning_rate = float(params["learning_rate_init"])
        batch_size = int(params["batch_size"])
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} parameters are malformed") from exc
    if params["backend"] != "sklearn" or params["device"] != "cpu" or int(params["max_epochs"]) != 1000 or int(params["patience"]) != 20:
        raise ContractError(f"{label} serialized protocol metadata is not frozen")
    if hidden not in COMPACT_GRID["hidden_layer_sizes"] or activation not in COMPACT_GRID["activation"] or alpha not in COMPACT_GRID["alpha"] or learning_rate not in COMPACT_GRID["learning_rate_init"] or batch_size not in COMPACT_GRID["batch_size"]:
        raise ContractError(f"{label} parameters are outside the frozen compact grid")


def _validate_results_text(directory: Path, prefix: str) -> None:
    text = (directory / "mlp_results.txt").read_text(encoding="utf-8")
    required = ["Evaluation unit: fov", f"Composition prefix: {prefix}", "MLP mode: nested_cv", "MLP backend: sklearn", "MLP device: cpu", "MLP max epochs: 1000", "MLP patience: 20", "MLP selection metric: weighted_f1", "MLP grid profile: compact", "MLP resampling: none", "MLP decision threshold: 0.5", "Skip SHAP: True", "--- SHAP skipped by configuration ---", "--- Final Performance Report ---"]
    missing = [value for value in required if value not in text]
    if missing or any(f"--- Processing Outer Fold {fold}/14 ---" not in text for fold in range(1, 15)):
        raise ContractError(f"{directory.name} mlp_results.txt is missing frozen protocol/fold metadata: {missing}")


def _group_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _validate_arm_artifacts(directory: Path, arm: str, prefix: str, composition_columns: list[str], enrichment_columns: list[str], niche_columns: list[str], predictions: pd.DataFrame, canonical: pd.Index, groups: pd.Series) -> None:
    _validate_results_text(directory, prefix)
    allowed = set(composition_columns) | set(enrichment_columns) | set(niche_columns)
    selected = pd.read_csv(directory / "selected_features_by_fold.csv")
    if len(selected) != 14 or selected.outer_fold.duplicated().any() or set(pd.to_numeric(selected.outer_fold, errors="raise").astype(int)) != set(range(1, 15)):
        raise ContractError(f"{arm} selected-feature report must contain exactly one row for folds 1..14")
    prediction_groups = predictions.groupby("outer_fold", sort=True)["test_group"].first().to_dict()
    for _, row in selected.iterrows():
        fold = int(row.outer_fold)
        features = [feature for feature in str(row.get("selected_features", "")).split("|") if feature]
        if int(row.selected_feature_count) != len(features) or len(features) != len(set(features)):
            raise ContractError(f"{arm} selected feature count/list is inconsistent")
        if not set(features).issubset(allowed):
            raise ContractError(f"{arm} selected feature is outside the allowed candidate union")
        if not set(composition_columns).issubset(features):
            raise ContractError(f"{arm} selected composition columns are absent from fold {fold}")
        enrichment_count = sum(feature in enrichment_columns for feature in features)
        niche_count = sum(feature in niche_columns for feature in features)
        if enrichment_count > 5 or niche_count > 20:
            raise ContractError(f"{arm} fold {fold} exceeds frozen candidate selection limits")
        expected_train_groups = sorted(set(groups.astype(str).unique()) - {str(prediction_groups[fold])})
        if _group_text(row.test_groups) != str(prediction_groups[fold]) or _group_text(row.train_groups) != ",".join(expected_train_groups):
            raise ContractError(f"{arm} selected/train/test group metadata disagrees for fold {fold}")
        fold_rows = predictions[predictions.outer_fold == fold]
        if int(row.test_rows) != len(fold_rows) or int(row.train_rows) != len(canonical) - len(fold_rows):
            raise ContractError(f"{arm} selected train/test row metadata disagrees for fold {fold}")
    best_payload = json.loads((directory / "best_params.json").read_text(encoding="utf-8"))
    best_records = best_payload.get("outer_folds") if isinstance(best_payload, dict) else None
    if not isinstance(best_records, list) or len(best_records) != 14:
        raise ContractError(f"{arm} best_params.json must contain exactly 14 outer fold records")
    seen = set()
    selected_by_fold = {int(row.outer_fold): row for _, row in selected.iterrows()}
    for record in best_records:
        fold = int(record.get("outer_fold", -1))
        if fold in seen or fold not in range(1, 15):
            raise ContractError(f"{arm} best_params.json has duplicate/invalid outer folds")
        seen.add(fold)
        expected_train_groups = sorted(set(groups.astype(str).unique()) - {str(prediction_groups[fold])})
        expected_features = [feature for feature in str(selected_by_fold[fold].selected_features).split("|") if feature]
        if _group_text(record.get("test_groups")) != str(prediction_groups[fold]) or sorted(str(item) for item in record.get("train_groups", [])) != expected_train_groups or int(record.get("test_rows", -1)) != int(selected_by_fold[fold].test_rows) or int(record.get("train_rows", -1)) != int(selected_by_fold[fold].train_rows) or int(record.get("selected_feature_count", -1)) != len(expected_features) or record.get("selected_features", []) != expected_features:
            raise ContractError(f"{arm} best_params fold metadata disagrees with evaluator artifacts")
        _validate_params(record.get("best_params", {}), f"{arm} outer fold {fold}")
    fixed = json.loads((directory / "fixed_params.json").read_text(encoding="utf-8"))
    if fixed.get("selection_scope") != "grouped_full_data" or fixed.get("selection_metric") != "weighted_f1" or fixed.get("grid_profile") != "compact" or fixed.get("resampling") != "none" or fixed.get("backend") != "sklearn" or fixed.get("device") != "cpu" or fixed.get("max_epochs") != 1000 or fixed.get("patience") != 20:
        raise ContractError(f"{arm} fixed_params.json protocol metadata is not frozen")
    _validate_params(fixed.get("best_params", {}), f"{arm} fixed params")
    labels = sorted(predictions.true_label.astype(str).unique())
    stored = pd.read_csv(directory / "confusion_matrix.csv", index_col=0)
    if stored.index.astype(str).tolist() != labels or stored.columns.astype(str).tolist() != labels or stored.shape != (len(labels), len(labels)):
        raise ContractError(f"{arm} stored confusion matrix labels/shape differ from predictions")
    expected = confusion_matrix(predictions.true_label.astype(str), predictions.predicted_label.astype(str), labels=labels)
    try:
        stored_values = stored.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{arm} confusion matrix is not numeric") from exc
    if not np.array_equal(stored_values, expected.astype(float)):
        raise ContractError(f"{arm} stored confusion matrix differs from recomputed pooled matrix")


def _feature_reports(directory: Path, arm: str, out: Path, composition_columns: list[str], n_folds: int = 14) -> None:
    selected = pd.read_csv(directory / "selected_features_by_fold.csv")
    if set(selected.outer_fold.astype(int)) != set(range(1, n_folds + 1)):
        raise ContractError(f"{arm} selected-feature report does not contain 14 folds")
    counts: dict[str, int] = {}
    for _, row in selected.iterrows():
        features = [feature for feature in str(row.get("selected_features", "")).split("|") if feature]
        if "selected_feature_count" in selected and int(row.selected_feature_count) != len(features):
            raise ContractError(f"{arm} selected_feature_count disagrees with selected_features")
        if not set(composition_columns).issubset(features):
            raise ContractError(f"{arm} selected composition columns are absent from an outer fold")
        if len(features) != len(set(features)):
            raise ContractError(f"{arm} selected features contain duplicates")
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
    rows = [{"arm": arm, "feature": feature, "fold_count": count, "fold_frequency": count / n_folds} for feature, count in sorted(counts.items())]
    pd.DataFrame(rows).to_csv(out / f"selected_feature_frequencies_{arm}.csv", index=False)
    family_predicates = {"composition": lambda x: x.startswith(("nmf_prop_", "novae_prop_")), "enrichment": lambda x: x.startswith("enrichment"), "niche_gene": lambda x: x.startswith("niche")}
    family_rows = []
    for family, predicate in family_predicates.items():
        features = [f for f in counts if predicate(f)]
        family_rows.append({"arm": arm, "family": family, "features_selected": len(features), "total_fold_selections": sum(counts[f] for f in features), "mean_fold_frequency": (sum(counts[f] for f in features) / len(features) / n_folds if features else 0.0)})
    unmatched = [f for f in counts if not any(predicate(f) for predicate in family_predicates.values())]
    family_rows.append({"arm": arm, "family": "other", "features_selected": len(unmatched), "total_fold_selections": sum(counts[f] for f in unmatched), "mean_fold_frequency": (sum(counts[f] for f in unmatched) / len(unmatched) / n_folds if unmatched else 0.0)})
    pd.DataFrame(family_rows).to_csv(out / f"family_stability_{arm}.csv", index=False)


def summarize(stage: Path, nmf: dict[str, Any], novae: dict[str, Any], input_hashes: dict[str, Any]) -> dict[str, Any]:
    canonical, targets, groups = nmf["combined"].index, nmf["targets"], nmf["groups"]
    n = _predictions(stage / "nmf", canonical, targets, groups)
    v = _predictions(stage / "novae", canonical, targets, groups)
    _validate_arm_artifacts(stage / "nmf", "nmf", "nmf_prop_", nmf["composition_columns"], [str(column) for column in nmf["enrichment"].columns], [str(column) for column in nmf["niche"].columns], n, canonical, groups)
    _validate_arm_artifacts(stage / "novae", "novae", "novae_prop_", novae["composition_columns"], [str(column) for column in novae["enrichment"].columns], [str(column) for column in novae["niche"].columns], v, canonical, groups)
    for col in ("item_id", "true_label", "test_group", "outer_fold", "decision_threshold", "positive_class"):
        if not n[col].astype(str).equals(v[col].astype(str)):
            raise ContractError(f"paired prediction alignment mismatch in {col}")
    labels = sorted(targets.unique().tolist())
    nmf_metrics, novae_metrics = _metrics(n.true_label, n.predicted_label, labels), _metrics(v.true_label, v.predicted_label, labels)
    pd.DataFrame(nmf_metrics["confusion_matrix"], index=labels, columns=labels).to_csv(stage / "confusion_matrix_nmf.csv")
    pd.DataFrame(novae_metrics["confusion_matrix"], index=labels, columns=labels).to_csv(stage / "confusion_matrix_novae.csv")
    joined = pd.DataFrame({"item_id": canonical.astype(str), "test_group": n.test_group.astype(str), "true_label": n.true_label.astype(str), "nmf_predicted_label": n.predicted_label.astype(str), "novae_predicted_label": v.predicted_label.astype(str)})
    joined["nmf_correct"] = joined.nmf_predicted_label == joined.true_label
    joined["novae_correct"] = joined.novae_predicted_label == joined.true_label
    joined.to_csv(stage / "paired_predictions.csv", index=False)
    patient_rows = []
    for patient, rows in joined.groupby("test_group", sort=True):
        nmf_correct, novae_correct = int(rows.nmf_correct.sum()), int(rows.novae_correct.sum())
        patient_rows.append({"patient": patient, "row_count": len(rows), "nmf_correct_count": nmf_correct, "novae_correct_count": novae_correct, "nmf_accuracy": nmf_correct / len(rows), "novae_accuracy": novae_correct / len(rows), "delta_accuracy_novae_minus_nmf": (novae_correct - nmf_correct) / len(rows)})
    patients = pd.DataFrame(patient_rows)
    patients.to_csv(stage / "per_patient_metrics.csv", index=False)
    counts = pd.crosstab(joined.nmf_correct, joined.novae_correct).reindex(index=[False, True], columns=[False, True], fill_value=0)
    counts.rename_axis(index="nmf_correct", columns="novae_correct").to_csv(stage / "paired_correctness_counts.csv")
    _feature_reports(stage / "nmf", "nmf", stage, nmf["composition_columns"])
    _feature_reports(stage / "novae", "novae", stage, novae["composition_columns"])
    summary = {"protocol": PROTOCOL, "arms": {"nmf": nmf_metrics, "novae": novae_metrics}, "delta_novae_minus_nmf": {key: novae_metrics[key] - nmf_metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}, "paired_correctness_counts": counts.astype(int).to_dict(), "warning": "NOVAE is exploratory reference=all; FOV rows are not independent; no patient/FOV significance or p-value claim is made.", "inputs": input_hashes}
    (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{ "metric": key, "nmf": nmf_metrics[key], "novae": novae_metrics[key], "delta_novae_minus_nmf": summary["delta_novae_minus_nmf"][key]} for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")]).to_csv(stage / "summary.csv", index=False)
    return summary


PROTOCOL = {"cohort": "historical_164", "unit": "fov", "mode": "nested_cv", "outer_cv": "patient_LOGO", "backend": "sklearn", "composition_arms": {"nmf": "nmf_prop_", "novae": "novae_prop_"}, "grid": "compact", "selection_metric": "weighted_f1", "resampling": "none", "threshold": 0.5, "max_epochs": 1000, "patience": 20, "top_enrichment": 5, "top_niche": 20, "seed": 42, "shap": False, "feature_selection": "training-fold-only mutual information", "independence_claim": False}


def run_comparison(*, nmf_feature_dir: Path = NMF_FEATURE_DIR, nmf_source_dir: Path = NMF_SOURCE_DIR, novae_feature_dir: Path = NOVAE_FEATURE_DIR, output_dir: Path, base_env: dict[str, str] | None = None, candidate_counts: dict[str, int] | None = None) -> dict[str, Any]:
    if output_dir.exists():
        raise ContractError(f"refusing existing final output: {output_dir}")
    nmf_feature_dir, nmf_source_dir, novae_feature_dir, output_dir = map(Path, (nmf_feature_dir, nmf_source_dir, novae_feature_dir, output_dir))
    _check_inherited_env(base_env if base_env is not None else None)
    nmf = preflight_arm(nmf_feature_dir, nmf_source_dir, prefix="nmf_prop_", candidate_counts=candidate_counts)
    nmf["composition_columns"] = [str(c) for c in nmf["composition_columns"]]
    novae_manifest = validate_novae_manifest(novae_feature_dir)
    novae = preflight_arm(novae_feature_dir, novae_feature_dir, prefix="novae_prop_", canonical=nmf["combined"].index, expected_targets=nmf["targets"], expected_groups=nmf["groups"], candidate_counts=candidate_counts)
    if not nmf["targets"].equals(novae["targets"]) or not nmf["groups"].equals(novae["groups"]):
        raise ContractError("arms do not have identical frozen targets/groups")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        (stage / "nmf").mkdir(); (stage / "novae").mkdir()
        envbase = dict(base_env or os.environ)
        envs = [_arm_env(envbase, nmf_feature_dir, nmf_source_dir, stage / "nmf", "nmf_prop_"), _arm_env(envbase, novae_feature_dir, novae_feature_dir, stage / "novae", "novae_prop_")]
        results = _run_arms(envs, [stage / "nmf", stage / "novae"])
        if any(code != 0 for code, _ in results):
            raise ContractError("one or both evaluator arms failed")
        nmf_enrichment = _table_path(nmf_source_dir / "enrichment_features_fov")
        nmf_niche = _table_path(nmf_source_dir / "niche_gene_features_fov")
        novae_enrichment = _table_path(novae_feature_dir / "enrichment_features_fov")
        novae_niche = _table_path(novae_feature_dir / "niche_gene_features_fov")
        input_hashes = {
            "evaluator": _hash(EVALUATOR),
            "nmf": _input_inventory([("combined_features_filtered.parquet", nmf_feature_dir / "combined_features_filtered.parquet"), ("targets_y.parquet", nmf_feature_dir / "targets_y.parquet"), ("groups.parquet", nmf_feature_dir / "groups.parquet"), ("enrichment_features_fov", nmf_enrichment), ("niche_gene_features_fov", nmf_niche), ("post_nmf_obs.csv", nmf_source_dir / "post_nmf_obs.csv")]),
            "novae": _input_inventory([("combined_features_filtered.parquet", novae_feature_dir / "combined_features_filtered.parquet"), ("targets_y.parquet", novae_feature_dir / "targets_y.parquet"), ("groups.parquet", novae_feature_dir / "groups.parquet"), ("enrichment_features_fov", novae_enrichment), ("niche_gene_features_fov", novae_niche), ("novae_feature_manifest.json", novae_feature_dir / "novae_feature_manifest.json")]),
            "novae_contract": {"contract": novae_manifest.get("contract"), "warning": novae_manifest.get("warning"), "provenance": novae_manifest.get("novae_pilot_provenance")},
        }
        summary = summarize(stage, nmf, novae, input_hashes)
        manifest = {"protocol": PROTOCOL, "selected_composition_columns": {"nmf": nmf["composition_columns"], "novae": novae["composition_columns"]}, "code_sha256": {"evaluator": _hash(EVALUATOR), "orchestrator": _hash(Path(__file__)), "launcher": _hash(REPO / "scripts" / "submit_novae_nmf_comparison.sh")}, "input_sha256": input_hashes, "summary": summary, "warning": "exploratory NOVAE reference=all; no independence/p-value claim"}
        manifest["output_sha256"] = _tree_hashes(stage)
        manifest["output_sha256_excludes"] = ["run_manifest.json"]
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise ContractError(f"refusing existing final output: {output_dir}")
        os.rename(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmf-feature-dir", type=Path, default=NMF_FEATURE_DIR)
    parser.add_argument("--nmf-source-dir", type=Path, default=NMF_SOURCE_DIR)
    parser.add_argument("--novae-feature-dir", type=Path, default=NOVAE_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_comparison(nmf_feature_dir=args.nmf_feature_dir, nmf_source_dir=args.nmf_source_dir, novae_feature_dir=args.novae_feature_dir, output_dir=args.output_dir)
    except Exception as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
