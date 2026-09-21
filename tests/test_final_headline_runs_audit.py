from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_final_headline_runs import AuditError, _parse_mlp_metadata, inspect_predictions, inspect_run, validate_source_metadata


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["f1", "f2", "f3", "f4"],
            "test_group": ["p1", "p1", "p2", "p2"],
            "outer_fold": [1, 1, 2, 2],
            "true_label": ["healthy", "healthy", "systemic_sclerosis", "systemic_sclerosis"],
            "predicted_label": ["healthy", "healthy", "systemic_sclerosis", "systemic_sclerosis"],
            "decision_threshold": [0.5, 0.5, 0.5, 0.5],
            "positive_class": ["systemic_sclerosis"] * 4,
            "positive_class_probability": [0.1, 0.4, 0.8, 0.9],
        }
    )


def test_source_metadata_patient_disease_and_fov_counts() -> None:
    result = validate_source_metadata(
        pd.DataFrame({"patient_id": ["p1", "p1", "p2"], "disease": ["healthy", "healthy", "ssc"], "fov_id": ["a", "b", "c"]}),
        expected_patients=2,
    )
    assert result["n_patients"] == 2
    assert result["n_fovs"] == 3


def test_source_metadata_rejects_patient_with_two_diseases() -> None:
    with pytest.raises(AuditError, match="single-valued"):
        validate_source_metadata(
            pd.DataFrame({"patient_id": ["p1", "p1"], "disease": ["healthy", "ssc"], "fov_id": ["a", "b"]}),
            expected_patients=1,
        )


def test_real_fold_prediction_headers_and_pooled_metrics() -> None:
    result = inspect_predictions(prediction_frame())
    assert result["technical_checks"] == {
        "unique_item_ids": True,
        "group_single_fold": True,
        "group_label_consistent": True,
        "finite_probabilities_0_1": True,
        "threshold_prediction_consistent": True,
    }
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("item_id", ["f1", "f1", "f3", "f4"], "not unique"),
        ("outer_fold", [1, 2, 2, 2], "confined"),
        ("positive_class_probability", [0.1, 1.2, 0.8, 0.9], r"in \[0,1\]"),
        ("predicted_label", ["healthy", "systemic_sclerosis", "systemic_sclerosis", "systemic_sclerosis"], "inconsistency"),
    ],
)
def test_prediction_failures(column: str, value: list[object], message: str) -> None:
    frame = prediction_frame()
    frame[column] = value
    with pytest.raises(AuditError, match=message):
        inspect_predictions(frame)


def test_prediction_labels_must_be_binary() -> None:
    frame = prediction_frame()
    frame.loc[0, "predicted_label"] = "other"
    with pytest.raises(AuditError, match="exactly binary"):
        inspect_predictions(frame)


def test_mlp_outer_fold_identities_are_unique_and_in_range(tmp_path: Path) -> None:
    path = tmp_path / "mlp_results.txt"
    path.write_text("MLP mode: nested_cv\nProcessing Outer Fold 1/2\nProcessing Outer Fold 1/2\n")
    with pytest.raises(AuditError, match="duplicate"):
        _parse_mlp_metadata(path)
    path.write_text("MLP mode: nested_cv\nProcessing Outer Fold 1/2\nProcessing Outer Fold 2/3\n")
    with pytest.raises(AuditError, match="inconsistent"):
        _parse_mlp_metadata(path)
    path.write_text("MLP mode: nested_cv\nProcessing Outer Fold 3/2\n")
    with pytest.raises(AuditError, match="outside declared range"):
        _parse_mlp_metadata(path)


def test_mlp_metadata_detects_incomplete_nested_cv(tmp_path: Path) -> None:
    run = tmp_path / "run" / "outputs" / "MLP_FOVFeatures_nested_cv"
    run.mkdir(parents=True)
    (run.parent.parent / "artifacts").mkdir()
    (run.parent.parent / "artifacts" / "manifest.json").write_text("{}")
    (run.parent.parent / "artifacts" / "run_summary.json").write_text("{}")
    (run.parent.parent / "post_nmf_features.csv").write_text("x\n1\n")
    (run / "mlp_results.txt").write_text(
        "MLP mode: nested_cv\nEvaluation unit: FOV\nOuter CV mode: LOGO\n"
        "Units: 44\nGroup count: 14\nProcessing Outer Fold 1/14\nProcessing Outer Fold 2/14\n"
    )
    result = inspect_run(tmp_path / "run" / "outputs", name="skin_1mm_split", expected_folds=14)
    metadata = result["artifacts"]["mlp_metadata"][0]
    assert metadata["mlp_mode"] == "nested_cv"
    assert metadata["outer_fold_count"] == 2
    assert result["nested_cv"][0]["incomplete"] is True


def test_modern_metadata_contract_and_group_derivation(tmp_path: Path) -> None:
    run = tmp_path / "run" / "outputs" / "MLP_FOVFeatures_eval"
    run.mkdir(parents=True)
    (run.parent.parent / "artifacts").mkdir()
    (run.parent.parent / "artifacts" / "manifest.json").write_text("{}")
    (run.parent.parent / "artifacts" / "run_summary.json").write_text("{}")
    (run.parent.parent / "post_nmf_features.csv").write_text("x\n1\n")
    (run / "mlp_results.txt").write_text(
        "MLP mode: nested_cv\nEvaluation unit: FOV\nOuter CV mode: LOGO\n"
        "Units: 44\nGroup count: 2\nProcessing Outer Fold 1/2\nProcessing Outer Fold 2/2\n"
        "--- Final Performance Report ---\n"
    )
    prediction_frame().to_csv(run / "fold_predictions.csv", index=False)
    result = inspect_run(tmp_path / "run" / "outputs", name="skin_1mm_split", expected_folds=2)
    output = result["artifacts"]["mlp_outputs"][0]
    assert output["metadata_contract"] == "modern_leakage_safe"
    assert output["path"].endswith("fold_predictions.csv")
    assert output["metadata_path"].endswith("mlp_results.txt")
    assert output["derived_item_count"] == 4
    assert output["derived_group_count"] == 2
    assert output["metrics"]["n_groups"] == 2
    assert result["nested_cv"][0]["incomplete"] is False


def test_render_only_launcher_is_quoted_and_cpu_only(tmp_path: Path) -> None:
    log_dir = tmp_path / "log-dir"
    job = tmp_path / "job-script.sbatch"
    env = os.environ.copy()
    env.update(
        AUDIT_LOG_DIR=str(log_dir),
        AUDIT_JOB_SCRIPT=str(job),
        AUDIT_OUTPUT_DIR=str(tmp_path / "audit output"),
        AUDIT_REPO_DIR=str(tmp_path / "repo dir"),
        AUDIT_SKIN_H5AD=str(tmp_path / "skin source.h5ad"),
    )
    script = Path("scripts/submit_final_headline_runs_audit.sh")
    subprocess.run([str(script), "--render-only"], check=True, env=env, text=True)
    text = job.read_text()
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=64gb" in text
    assert "--gres" not in text and "--gpus" not in text
    assert "SKIN_H5AD='/" in text
    assert 'python scripts/audit_final_headline_runs.py \\' in text
    subprocess.run(["bash", "-n", str(job)], check=True)
    bad_env = dict(env)
    bad_env["AUDIT_LOG_DIR"] = str(tmp_path / "log dir rejected")
    bad = subprocess.run([str(script), "--render-only"], env=bad_env, text=True, capture_output=True)
    assert bad.returncode == 2
    assert "whitespace" in bad.stderr


@pytest.mark.parametrize("variable", ["AUDIT_ACCOUNT", "AUDIT_QOS", "AUDIT_PARTITION"])
def test_render_rejects_whitespace_in_scheduler_values(tmp_path: Path, variable: str) -> None:
    env = os.environ.copy()
    env.update(
        AUDIT_LOG_DIR=str(tmp_path / "logs"),
        AUDIT_JOB_SCRIPT=str(tmp_path / "job.sbatch"),
        AUDIT_OUTPUT_DIR=str(tmp_path / "audit output"),
        AUDIT_REPO_DIR=str(tmp_path / "repo dir"),
        AUDIT_SKIN_H5AD=str(tmp_path / "skin source.h5ad"),
        **{variable: "unsafe value"},
    )
    result = subprocess.run(["scripts/submit_final_headline_runs_audit.sh", "--render-only"], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "whitespace" in result.stderr


def test_atomic_failure_leaves_no_output(tmp_path: Path) -> None:
    # Importing the parser keeps this test independent of anndata and real H5AD files.
    from scripts.audit_final_headline_runs import build_parser, run_audit

    output = tmp_path / "audit"
    args = build_parser().parse_args(["--output-dir", str(output)])
    with pytest.raises(AuditError):
        run_audit(args)
    assert not output.exists()
    assert not list(tmp_path.glob(".audit.*.partial"))
