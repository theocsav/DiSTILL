"""Synthetic contract tests for the historical-164 paired orchestrator."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
from scripts import run_novae_nmf_comparison as comparison


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_arm(directory: Path, *, prefix: str, index: pd.Index, labels: pd.Series, groups: pd.Series, legacy: bool = False) -> None:
    directory.mkdir(parents=True)
    values = np.arange(len(index), dtype=float)
    if prefix == "novae_prop_":
        columns = {f"novae_prop_L{i}": values * 0 + (1.0 / 9.0) for i in range(9)}
    else:
        columns = {f"nmf_prop_{i}": values * 0 + (1.0 / 9.0) for i in range(9)}
    if legacy:
        columns["legacy_global_selected"] = values
    pd.DataFrame(columns, index=index).to_parquet(directory / "combined_features_filtered.parquet")
    labels.to_frame("label").to_parquet(directory / "targets_y.parquet")
    groups.to_frame("group").to_parquet(directory / "groups.parquet")
    pd.DataFrame({"enrichment_x": values}, index=index).to_parquet(directory / "enrichment_features_fov.parquet")
    pd.DataFrame({"niche_gene_x": values}, index=index).to_parquet(directory / "niche_gene_features_fov.parquet")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, pd.Index]:
    counts = [13] + [12] * 8 + [11] * 5
    patient_values = [f"P{patient:02d}" for patient, count in enumerate(counts) for _ in range(count)]
    index = pd.Index([f"{patient}_F{i:03d}" for i, patient in enumerate(patient_values)], name="fov_key")
    groups = pd.Series(patient_values, index=index)
    labels = groups.map(lambda patient: "healthy" if int(patient[1:]) < 5 else "systemic_sclerosis")
    nmf, source, novae = tmp_path / "nmf_features", tmp_path / "nmf_source", tmp_path / "novae_features"
    _write_arm(nmf, prefix="nmf_prop_", index=index, labels=labels, groups=groups, legacy=True)
    _write_arm(novae, prefix="novae_prop_", index=index, labels=labels, groups=groups)
    source.mkdir()
    pd.DataFrame({"field_of_view": index.to_numpy(), "patient": groups.to_numpy(), "Disease_State": labels.to_numpy(), "NMF_factor": [i % 9 for i in range(164)]}).to_csv(source / "post_nmf_obs.csv")
    pd.DataFrame({"enrichment_x": np.arange(164.)}, index=index).to_parquet(source / "enrichment_features_fov.parquet")
    pd.DataFrame({"niche_gene_x": np.arange(164.)}, index=index).to_parquet(source / "niche_gene_features_fov.parquet")
    outputs = {path.name: {"sha256": _hash(path), "path": str(path)} for path in sorted(novae.iterdir())}
    manifest = {"contract": "historical_164", "warning": "exploratory reference=all; not confirmatory", "novae_pilot_provenance": {"analysis_scope": "exploratory", "reference": "all", "inference_mode": "zero_shot", "dataset_id": "skin_visium_ssc_paired_cpu_calibrated", "coordinate_strategy": "visium_explicit_scale", "primary_resolution": 1.0, "domain_key": "novae_domains_res1.0", "neighborhood_valid_key": "neighborhood_valid", "accelerator": "cpu", "device": "cpu", "workers": 0, "seed": 42, "confirmatory_held_out_classification_allowed": False, "minimum_domain_assignment_coverage": 0.70, "input_sha256": "262418e8e7ed06de805e940406f3ae9e41487ce085da1ae8f940c81f95daf6dd", "checkpoint_sha256": "1422f9f72d6e532921bf8a90f0996f1c46c6891f6ecbc73e404521ec5aa7b04a", "deterministic_policy": {"requested": True, "effective": True}}, "outputs": outputs}
    (novae / "novae_feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return nmf, source, novae, index


def _mock_evaluator_output(directory: Path, *, prefix: str, index: pd.Index, labels: pd.Series, groups: pd.Series) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    comp = [f"novae_prop_L{i}" for i in range(9)] if prefix == "novae_prop_" else [f"nmf_prop_{i}" for i in range(9)]
    selected_rows, best_rows, predictions = [], [], []
    unique_groups = sorted(groups.unique())
    params = {"hidden_layer_sizes": [16], "activation": "relu", "alpha": 0.001, "learning_rate_init": 0.001, "batch_size": 8, "backend": "sklearn", "device": "cpu", "max_epochs": 1000, "patience": 20}
    for fold, group in enumerate(unique_groups, 1):
        mask = groups.astype(str) == group
        test_rows = int(mask.sum())
        features = comp + ["enrichment_x", "niche_gene_x"]
        selected_rows.append({"outer_fold": fold, "train_groups": ",".join(g for g in unique_groups if g != group), "test_groups": group, "train_rows": len(index) - test_rows, "test_rows": test_rows, "selected_feature_count": len(features), "selected_features": "|".join(features)})
        best_rows.append({"outer_fold": fold, "train_groups": [g for g in unique_groups if g != group], "test_groups": [group], "train_rows": len(index) - test_rows, "test_rows": test_rows, "selected_feature_count": len(features), "selected_features": features, "best_params": params})
        for item_id, label in zip(index[mask], labels[mask], strict=True):
            predictions.append({"outer_fold": fold, "item_id": item_id, "test_group": group, "true_label": label, "predicted_label": label, "decision_threshold": 0.5, "positive_class": "systemic_sclerosis", "positive_class_probability": 0.75 if label == "systemic_sclerosis" else 0.25})
    pd.DataFrame(predictions).to_csv(directory / "fold_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(directory / "selected_features_by_fold.csv", index=False)
    (directory / "best_params.json").write_text(json.dumps({"outer_folds": best_rows}), encoding="utf-8")
    (directory / "fixed_params.json").write_text(json.dumps({"selection_scope": "grouped_full_data", "selection_metric": "weighted_f1", "grid_profile": "compact", "resampling": "none", "backend": "sklearn", "device": "cpu", "max_epochs": 1000, "patience": 20, "best_params": params}), encoding="utf-8")
    pd.DataFrame([[61, 0], [0, 103]], index=["healthy", "systemic_sclerosis"], columns=["healthy", "systemic_sclerosis"]).to_csv(directory / "confusion_matrix.csv")
    text = ["Evaluation unit: fov", f"Composition prefix: {prefix}", "MLP mode: nested_cv", "MLP backend: sklearn", "MLP device: cpu", "MLP max epochs: 1000", "MLP patience: 20", "MLP selection metric: weighted_f1", "MLP grid profile: compact", "MLP resampling: none", "MLP decision threshold: 0.5", "Skip SHAP: True", "--- SHAP skipped by configuration ---"] + [f"--- Processing Outer Fold {fold}/14 ---" for fold in range(1, 15)] + ["--- Final Performance Report ---"]
    (directory / "mlp_results.txt").write_text("\n".join(text), encoding="utf-8")


def test_authoritative_mapping_integer_csv_index_preserves_values(tmp_path: Path) -> None:
    _, source, _, _ = _fixture(tmp_path)
    mapping = comparison._authoritative_mapping(source)
    assert mapping["fov"].notna().all() and mapping["patient"].notna().all() and mapping["label"].notna().all()


def test_preflight_ignores_legacy_nmf_columns_and_rejects_alias(tmp_path: Path) -> None:
    nmf, source, _, _ = _fixture(tmp_path)
    small = {"enrichment": 1, "niche": 1}
    checked = comparison.preflight_arm(nmf, source, prefix="nmf_prop_", candidate_counts=small)
    assert checked["composition_columns"] == [f"nmf_prop_{i}" for i in range(9)]
    combined = pd.read_parquet(nmf / "combined_features_filtered.parquet")
    combined["novae_prop_L0"] = 0.0
    combined.to_parquet(nmf / "combined_features_filtered.parquet")
    with pytest.raises(comparison.ContractError, match="forbidden"):
        comparison.preflight_arm(nmf, source, prefix="nmf_prop_", candidate_counts=small)


def test_postflight_rejects_mutated_evaluator_artifacts(tmp_path: Path) -> None:
    nmf, _, _, index = _fixture(tmp_path)
    labels = pd.read_parquet(nmf / "targets_y.parquet").iloc[:, 0].astype(str)
    groups = pd.read_parquet(nmf / "groups.parquet").iloc[:, 0].astype(str)
    output = tmp_path / "arm"
    _mock_evaluator_output(output, prefix="nmf_prop_", index=index, labels=labels, groups=groups)
    predictions = comparison._predictions(output, index, labels, groups)
    allowed = [f"nmf_prop_{i}" for i in range(9)]
    kwargs = {"directory": output, "arm": "nmf", "prefix": "nmf_prop_", "composition_columns": allowed, "enrichment_columns": ["enrichment_x"], "niche_columns": ["niche_gene_x"], "predictions": predictions, "canonical": index, "groups": groups}
    comparison._validate_arm_artifacts(**kwargs)
    selected = pd.read_csv(output / "selected_features_by_fold.csv")
    pd.concat([selected, selected.iloc[[0]]], ignore_index=True).to_csv(output / "selected_features_by_fold.csv", index=False)
    with pytest.raises(comparison.ContractError, match="exactly one row"):
        comparison._validate_arm_artifacts(**kwargs)
    _mock_evaluator_output(output, prefix="nmf_prop_", index=index, labels=labels, groups=groups)
    selected = pd.read_csv(output / "selected_features_by_fold.csv")
    selected.loc[0, "selected_features"] += "|unknown_feature"
    selected.loc[0, "selected_feature_count"] += 1
    selected.to_csv(output / "selected_features_by_fold.csv", index=False)
    with pytest.raises(comparison.ContractError, match="allowed candidate"):
        comparison._validate_arm_artifacts(**kwargs)
    _mock_evaluator_output(output, prefix="nmf_prop_", index=index, labels=labels, groups=groups)
    fixed = json.loads((output / "fixed_params.json").read_text())
    fixed["best_params"]["activation"] = "bad"
    (output / "fixed_params.json").write_text(json.dumps(fixed))
    with pytest.raises(comparison.ContractError, match="compact grid"):
        comparison._validate_arm_artifacts(**kwargs)
    _mock_evaluator_output(output, prefix="nmf_prop_", index=index, labels=labels, groups=groups)
    (output / "mlp_results.txt").write_text("MLP mode: evaluate_fixed")
    with pytest.raises(comparison.ContractError, match="protocol/fold"):
        comparison._validate_arm_artifacts(**kwargs)
    _mock_evaluator_output(output, prefix="nmf_prop_", index=index, labels=labels, groups=groups)
    confusion = pd.read_csv(output / "confusion_matrix.csv", index_col=0)
    confusion.iloc[0, 0] = 0
    confusion.to_csv(output / "confusion_matrix.csv")
    with pytest.raises(comparison.ContractError, match="confusion matrix"):
        comparison._validate_arm_artifacts(**kwargs)


def test_successful_two_arm_summarize_publishes_metrics_and_artifacts(tmp_path: Path) -> None:
    nmf_dir, source, novae_dir, index = _fixture(tmp_path)
    labels = pd.read_parquet(nmf_dir / "targets_y.parquet").iloc[:, 0].astype(str)
    groups = pd.read_parquet(nmf_dir / "groups.parquet").iloc[:, 0].astype(str)
    small = {"enrichment": 1, "niche": 1}
    nmf = comparison.preflight_arm(nmf_dir, source, prefix="nmf_prop_", candidate_counts=small)
    novae = comparison.preflight_arm(novae_dir, novae_dir, prefix="novae_prop_", canonical=index, expected_targets=labels, expected_groups=groups, candidate_counts=small)
    stage = tmp_path / "stage"
    _mock_evaluator_output(stage / "nmf", prefix="nmf_prop_", index=index, labels=labels, groups=groups)
    _mock_evaluator_output(stage / "novae", prefix="novae_prop_", index=index, labels=labels, groups=groups)
    summary = comparison.summarize(stage, nmf, novae, {"synthetic": True})
    assert summary["arms"]["nmf"]["accuracy"] == 1.0
    assert summary["arms"]["novae"]["accuracy"] == 1.0
    assert (stage / "summary.json").is_file() and (stage / "summary.csv").is_file()
    assert (stage / "paired_predictions.csv").is_file() and (stage / "per_patient_metrics.csv").is_file()
    assert (stage / "confusion_matrix_nmf.csv").is_file() and (stage / "confusion_matrix_novae.csv").is_file()
    assert (stage / "selected_feature_frequencies_novae.csv").is_file()
    selected_novae = pd.read_csv(stage / "novae" / "selected_features_by_fold.csv")
    assert all("novae_prop_L0" in value.split("|") for value in selected_novae.selected_features)
    assert all(not any(f"nmf_prop_{i}" in value.split("|") for i in range(9)) for value in selected_novae.selected_features)


def test_prediction_alignment_and_metrics(tmp_path: Path) -> None:
    nmf, _, _, index = _fixture(tmp_path)
    labels = pd.read_parquet(nmf / "targets_y.parquet").iloc[:, 0].astype(str)
    groups = pd.read_parquet(nmf / "groups.parquet").iloc[:, 0].astype(str)
    output = tmp_path / "arm"
    _mock_evaluator_output(output, prefix="nmf_prop_", index=index, labels=labels, groups=groups)
    predictions = comparison._predictions(output, index, labels, groups)
    assert comparison._metrics(predictions.true_label, predictions.predicted_label, ["healthy", "systemic_sclerosis"])["accuracy"] == 1.0


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX bash")
def test_launcher_render_and_safety(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    script = root / "scripts" / "submit_novae_nmf_comparison.sh"
    env = {**os.environ, "NOVAE_COMPARISON_REPO_DIR": str(root), "NOVAE_COMPARISON_RUN_ROOT": str(tmp_path / "run"), "NOVAE_COMPARISON_OUTPUT_DIR": str(tmp_path / "run" / "result"), "NOVAE_COMPARISON_JOB_SCRIPT": str(tmp_path / "run" / "job.sbatch"), "NOVAE_COMPARISON_LOG_DIR": str(tmp_path / "run" / "logs")}
    subprocess.run(["bash", str(script), "--render-only"], env=env, check=True, capture_output=True, text=True)
    text = (tmp_path / "run" / "job.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=2" in text and "#SBATCH --mem=96gb" in text and "ibd_cosmx_k4" in text and 'CUDA_VISIBLE_DEVICES=""' in text
    assert subprocess.run(["bash", str(script), "--render-only"], env={**env, "NICHERUNNER_MLP_MODE": "evaluate_fixed"}, capture_output=True).returncode == 2


def test_atomic_failure_cleans_staging_and_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nmf, source, novae, _ = _fixture(tmp_path)
    existing = tmp_path / "final"
    existing.mkdir()
    small = {"enrichment": 1, "niche": 1}
    with pytest.raises(comparison.ContractError, match="existing final"):
        comparison.run_comparison(nmf_feature_dir=nmf, nmf_source_dir=source, novae_feature_dir=novae, output_dir=existing, base_env={}, candidate_counts=small)
    existing.rmdir()
    monkeypatch.setattr(comparison, "_run_arms", lambda *_args, **_kwargs: [(1, "mock failure"), (0, "")])
    with pytest.raises(comparison.ContractError, match="evaluator arms failed"):
        comparison.run_comparison(nmf_feature_dir=nmf, nmf_source_dir=source, novae_feature_dir=novae, output_dir=existing, base_env={}, candidate_counts=small)
    assert not existing.exists() and not list(tmp_path.glob(f".{existing.name}.*"))
