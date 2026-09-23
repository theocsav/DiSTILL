"""Synthetic contract tests for the predeclared classifier ablation."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts import run_novae_classifier_ablation as ablation


def test_predeclared_configuration_matrix_and_exact_environment() -> None:
    assert ablation.ABLATIONS == {
        "composition_only": (0, 0),
        "composition_enrichment": (5, 0),
        "composition_niche": (0, 20),
        "full": (5, 20),
    }
    for name, (enrichment, niche) in ablation.ABLATIONS.items():
        protocol = ablation.protocol_for(name)
        assert protocol["top_enrichment"] == enrichment
        assert protocol["top_niche"] == niche
        env = ablation.expected_environment(name, "nmf_prop_")
        assert env["NICHERUNNER_TOP_ENRICHMENT_FEATURES"] == str(enrichment)
        assert env["NICHERUNNER_TOP_NICHE_GENE_FEATURES"] == str(niche)
        assert env["NICHERUNNER_MLP_MODE"] == "nested_cv"
        assert env["NICHERUNNER_SKIP_SHAP"] == "1"


def test_protocol_and_environment_mismatch_is_rejected() -> None:
    with pytest.raises(ablation.ContractError, match="unknown ablation"):
        ablation.protocol_for("performance_selected")
    with pytest.raises(ablation.ContractError, match="inherited"):
        ablation._check_inherited_env({"NICHERUNNER_MLP_MODE": "evaluate_fixed"})


def test_resolved_collision_and_unused_full_entry_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ablation.ContractError, match="collides"):
        ablation.run_ablation(nmf_feature_dir=source, nmf_source_dir=source, novae_feature_dir=source, full_primary_dir=source, output_root=source / "nested", base_env={})
    output = tmp_path / "output"
    (output / "full").mkdir(parents=True)
    with pytest.raises(ablation.ContractError, match="unexpected"):
        ablation.run_ablation(nmf_feature_dir=tmp_path / "nmf", nmf_source_dir=tmp_path / "source2", novae_feature_dir=tmp_path / "novae", full_primary_dir=tmp_path / "primary", output_root=output, base_env={})
    stale = tmp_path / "stale"
    (stale / ".composition_only.partial").mkdir(parents=True)
    with pytest.raises(ablation.ContractError, match="unexpected"):
        ablation.run_ablation(nmf_feature_dir=tmp_path / "nmf", nmf_source_dir=tmp_path / "source2", novae_feature_dir=tmp_path / "novae", full_primary_dir=tmp_path / "primary", output_root=stale, base_env={})


def test_exact_feature_family_counts_and_paired_alignment() -> None:
    import pandas as pd

    ablation._require_exact_family_counts(["c", "e1", "e2", "n1"], ["e1", "e2"], ["n1"], 2, 1, "synthetic")
    with pytest.raises(ablation.ContractError, match="exactly"):
        ablation._require_exact_family_counts(["c", "e1", "n1"], ["e1", "e2"], ["n1"], 2, 1, "synthetic")
    nmf = pd.DataFrame({"item_id": ["a"], "true_label": ["h"], "test_group": ["p"], "outer_fold": [1], "decision_threshold": [0.5], "positive_class": ["s"]})
    novae = nmf.copy()
    novae.loc[0, "test_group"] = "other"
    with pytest.raises(ablation.ContractError, match="test_group"):
        ablation._validate_paired_prediction_columns(nmf, novae, "synthetic")


def test_primary_protocol_does_not_invent_a_postflight_key() -> None:
    assert set(ablation.primary.PROTOCOL) == {
        "cohort", "unit", "mode", "outer_cv", "backend", "composition_arms", "grid",
        "selection_metric", "resampling", "threshold", "max_epochs", "patience",
        "top_enrichment", "top_niche", "seed", "shap", "feature_selection", "independence_claim",
    }


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX bash")
def test_launcher_render_safety_and_resources(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    full = tmp_path / "full"
    full.mkdir()
    run_root = tmp_path / "run"
    env = {
        **os.environ,
        "NOVAE_CLASSIFIER_ABLATION_REPO_DIR": str(root),
        "NOVAE_CLASSIFIER_ABLATION_FULL_PRIMARY_DIR": str(full),
        "NOVAE_CLASSIFIER_ABLATION_RUN_ROOT": str(run_root),
        "NOVAE_CLASSIFIER_ABLATION_OUTPUT_ROOT": str(run_root / "output"),
        "NOVAE_CLASSIFIER_ABLATION_JOB_SCRIPT": str(run_root / "job.sbatch"),
        "NOVAE_CLASSIFIER_ABLATION_LOG_DIR": str(run_root / "logs"),
    }
    script = root / "scripts" / "submit_novae_classifier_ablation.sh"
    subprocess.run(["bash", str(script), "--render-only"], env=env, check=True, capture_output=True, text=True)
    rendered = (run_root / "job.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=2" in rendered
    assert "#SBATCH --mem=96gb" in rendered
    assert "#SBATCH --time=24:00:00" in rendered
    assert 'CUDA_VISIBLE_DEVICES=""' in rendered
    assert "run_novae_classifier_ablation.py" in rendered
    default_run = tmp_path / "default-run"
    default_env = {key: value for key, value in env.items() if key != "NOVAE_CLASSIFIER_ABLATION_FULL_PRIMARY_DIR"}
    default_env.update({"NOVAE_CLASSIFIER_ABLATION_RUN_ROOT": str(default_run), "NOVAE_CLASSIFIER_ABLATION_OUTPUT_ROOT": str(default_run / "output"), "NOVAE_CLASSIFIER_ABLATION_JOB_SCRIPT": str(default_run / "job.sbatch"), "NOVAE_CLASSIFIER_ABLATION_LOG_DIR": str(default_run / "logs")})
    subprocess.run(["bash", str(script), "--render-only"], env=default_env, check=True, capture_output=True, text=True)
    default_job = (default_run / "job.sbatch").read_text(encoding="utf-8")
    assert f"{root}/runs/novae_nmf_comparison_20260922T214308Z_1310236/historical164_nmf_vs_novae" in default_job
    existing_output = run_root / "resumable-output"
    existing_output.mkdir()
    resume_env = {**env, "NOVAE_CLASSIFIER_ABLATION_OUTPUT_ROOT": str(existing_output), "NOVAE_CLASSIFIER_ABLATION_JOB_SCRIPT": str(run_root / "resume-job.sbatch")}
    subprocess.run(["bash", str(script), "--render-only"], env=resume_env, check=True, capture_output=True, text=True)
    assert (run_root / "resume-job.sbatch").is_file()
    failed = subprocess.run(["bash", str(script), "--render-only"], env={**env, "NICHERUNNER_MLP_MODE": "evaluate_fixed"}, capture_output=True)
    assert failed.returncode == 2


def test_publish_aggregate_end_to_end_mocked(tmp_path: Path) -> None:
    import json
    import pandas as pd

    root = tmp_path / "root"
    root.mkdir()
    full = tmp_path / "full"
    for arm, prefix in (("nmf", "nmf_prop_"), ("novae", "novae_prop_")):
        arm_dir = full / arm
        arm_dir.mkdir(parents=True)
        pd.DataFrame({"outer_fold": range(1, 15), "selected_features": [f"{prefix}0" for _ in range(14)]}).to_csv(arm_dir / "selected_features_by_fold.csv", index=False)
    (full / "run_manifest.json").write_text("{}", encoding="utf-8")
    n = pd.DataFrame({"item_id": ["a", "b"], "test_group": ["p1", "p2"], "true_label": ["healthy", "systemic_sclerosis"], "predicted_label": ["healthy", "systemic_sclerosis"]})
    v = n.copy()
    v.loc[0, "predicted_label"] = "systemic_sclerosis"
    metrics = ablation.primary._metrics(n.true_label, n.predicted_label, ["healthy", "systemic_sclerosis"])
    v_metrics = ablation.primary._metrics(v.true_label, v.predicted_label, ["healthy", "systemic_sclerosis"])
    config = root / "composition_only"
    config.mkdir()
    json_summary = {"ablation": "composition_only", "arms": {"nmf": metrics, "novae": v_metrics}, "delta_novae_minus_nmf": {key: v_metrics[key] - metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}}
    (config / "summary.json").write_text(json.dumps(json_summary), encoding="utf-8")
    n.to_csv(config / "paired_predictions.csv", index=False)
    pd.DataFrame({"ablation": ["composition_only"], "patient": ["p1"], "row_count": [1], "nmf_accuracy": [1.0], "novae_accuracy": [0.0], "delta_accuracy_novae_minus_nmf": [-1.0]}).to_csv(config / "per_patient_metrics.csv", index=False)
    pd.DataFrame({"ablation": ["composition_only"], "arm": ["nmf"], "feature": ["nmf_prop_0"], "fold_count": [14], "fold_frequency": [1.0]}).to_csv(config / "feature_stability.csv", index=False)
    nmf = {"targets": pd.Series(["healthy", "systemic_sclerosis"]), "composition_columns": ["nmf_prop_0"]}
    novae = {"composition_columns": ["novae_prop_0"]}
    result = ablation.publish_aggregate(root, full, n, v, ["composition_only"], nmf, novae, {"synthetic": True})
    assert result["full_primary_unchanged"] is True
    assert (root / "aggregate" / "pooled_metrics_long.csv").is_file()
    assert (root / "aggregate" / "feature_stability_long.csv").is_file()


def test_run_ablation_mocked_reaches_atomic_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import pandas as pd

    index = pd.Index(["a", "b"])
    labels = pd.Series(["healthy", "systemic_sclerosis"], index=index)
    groups = pd.Series(["p1", "p2"], index=index)
    n = pd.DataFrame({"item_id": ["a", "b"], "outer_fold": [1, 2], "test_group": ["p1", "p2"], "true_label": labels.tolist(), "predicted_label": labels.tolist(), "decision_threshold": [0.5, 0.5], "positive_class": ["systemic_sclerosis"] * 2})
    v = n.copy()
    source = {"combined": pd.DataFrame(index=index), "targets": labels, "groups": groups, "composition_columns": ["nmf_prop_0"], "enrichment": pd.DataFrame(columns=["e1"]), "niche": pd.DataFrame(columns=["n1"])}
    novae_source = {**source, "composition_columns": ["novae_prop_0"]}
    monkeypatch.setattr(ablation.primary, "preflight_arm", lambda *args, prefix, **kwargs: source if prefix == "nmf_prop_" else novae_source)
    monkeypatch.setattr(ablation.primary, "validate_novae_manifest", lambda path: {"contract": "historical_164", "warning": "reference=all exploratory", "novae_pilot_provenance": {}})
    monkeypatch.setattr(ablation, "_expected_input_hashes", lambda *args: {"synthetic": True})
    full = tmp_path / "full"
    for arm, prefix in (("nmf", "nmf_prop_"), ("novae", "novae_prop_")):
        (full / arm).mkdir(parents=True)
        pd.DataFrame({"outer_fold": range(1, 15), "selected_features": [f"{prefix}0"] * 14}).to_csv(full / arm / "selected_features_by_fold.csv", index=False)
    (full / "run_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ablation, "validate_full_primary", lambda *args: (n, v, {}))
    monkeypatch.setattr(ablation, "_run_arms", lambda envs, outputs: [(0, ""), (0, "")])
    def fake_summary(directory, ablation_name, nmf_source, novae_source, input_hashes):
        metrics = ablation.primary._metrics(n.true_label, n.predicted_label, ["healthy", "systemic_sclerosis"])
        summary = {"ablation": ablation_name, "arms": {"nmf": metrics, "novae": metrics}, "delta_novae_minus_nmf": {key: 0.0 for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}}
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        n.to_csv(directory / "paired_predictions.csv", index=False)
        pd.DataFrame({"ablation": [ablation_name], "patient": ["p1"], "row_count": [1], "nmf_accuracy": [1.0], "novae_accuracy": [1.0], "delta_accuracy_novae_minus_nmf": [0.0]}).to_csv(directory / "per_patient_metrics.csv", index=False)
        pd.DataFrame({"ablation": [ablation_name], "arm": ["nmf"], "feature": ["nmf_prop_0"], "fold_count": [14], "fold_frequency": [1.0]}).to_csv(directory / "feature_stability.csv", index=False)
        return summary
    monkeypatch.setattr(ablation, "summarize_config", fake_summary)
    result = ablation.run_ablation(nmf_feature_dir=tmp_path / "nmf", nmf_source_dir=tmp_path / "source", novae_feature_dir=tmp_path / "novae", full_primary_dir=full, output_root=tmp_path / "output", base_env={})
    assert result["full_primary_unchanged"] is True
    assert (tmp_path / "output" / "aggregate" / "run_manifest.json").is_file()


def test_long_form_metrics_are_representation_by_ablation() -> None:
    import pandas as pd

    n = pd.DataFrame({"true_label": ["healthy", "systemic_sclerosis"], "predicted_label": ["healthy", "systemic_sclerosis"]})
    v = pd.DataFrame({"true_label": ["healthy", "systemic_sclerosis"], "predicted_label": ["systemic_sclerosis", "systemic_sclerosis"]})
    matrix = ablation._metrics_long("composition_only", n, v, ["healthy", "systemic_sclerosis"])
    assert set(matrix.representation) == {"nmf", "novae"}
    assert len(matrix) == 8
    assert set(matrix.metric) == {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"}
    assert not matrix.loc[matrix.metric == "accuracy", "delta_novae_minus_nmf"].eq(0.0).any()
