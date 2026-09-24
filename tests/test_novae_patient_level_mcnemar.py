"""Synthetic contract tests for the exact patient-level McNemar report."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_novae_patient_level_mcnemar as report


def patient_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = ["healthy"] * 4 + ["systemic_sclerosis"] * 10
    counts = [16, 15, 15, 15] + [10] * 9 + [13]
    for configuration in report.CONFIGURATIONS:
        for arm in report.ARMS:
            for method in report.METHODS:
                for fold, (label, fov_count) in enumerate(zip(labels, counts, strict=True), 1):
                    rows.append(
                        {
                            "configuration": configuration,
                            "arm": arm,
                            "method": method,
                            "patient": f"patient-{fold:02d}",
                            "outer_fold": fold,
                            "fov_count": fov_count,
                            "score": 0.75 if label == "systemic_sclerosis" else 0.25,
                            "vote_fraction": 1.0 if label == "systemic_sclerosis" else 0.0,
                            "tie": False,
                            "true_label": label,
                            "predicted_label": label,
                            "correct": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_exact_mcnemar_orientations_and_zero() -> None:
    def result(nmf: list[bool], novae: list[bool]) -> dict[str, object]:
        return report.exact_mcnemar(pd.DataFrame({"nmf_correct": nmf, "novae_correct": novae}), "full", "primary_mean")

    one_vs_zero = result([True], [False])
    assert one_vs_zero["nmf_only_correct"] == 1
    assert one_vs_zero["novae_only_correct"] == 0
    assert one_vs_zero["discordant"] == 1
    assert one_vs_zero["exact_two_sided_p"] == pytest.approx(1.0)
    reverse = result([False], [True])
    assert reverse["nmf_only_correct"] == 0
    assert reverse["novae_only_correct"] == 1
    assert reverse["exact_two_sided_p"] == pytest.approx(1.0)
    balanced = result([True, False], [False, True])
    assert balanced["nmf_only_correct"] == balanced["novae_only_correct"] == 1
    assert balanced["exact_two_sided_p"] == pytest.approx(1.0)
    zero = result([True, False], [True, False])
    assert zero["discordant"] == 0
    assert zero["exact_two_sided_p"] == 1.0


def test_validation_recomputes_booleans_alignment_and_duplicates() -> None:
    frame = report.validate_patient_predictions(patient_table())
    assert len(frame) == report.EXPECTED_ROWS
    bad = patient_table()
    bad.loc[0, "correct"] = False
    with pytest.raises(report.ContractError, match="correct"):
        report.validate_patient_predictions(bad)
    bad = patient_table()
    bad.loc[0, "outer_fold"] = 2
    with pytest.raises(report.ContractError, match="alignment"):
        report.validate_patient_predictions(bad)
    bad = patient_table().iloc[:-1].copy()
    bad = pd.concat([bad, patient_table().iloc[[0]]], ignore_index=True)
    with pytest.raises(report.ContractError, match="duplicate"):
        report.validate_patient_predictions(bad)


def _write_input(root: Path, frame: pd.DataFrame) -> None:
    root.mkdir()
    csv = root / "patient_predictions_long.csv"
    frame.to_csv(csv, index=False)
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    protocol = {
        "cohort": "historical-164",
        "unit": "FOV predictions pooled to patient",
        "threshold": 0.5,
        "positive_class": "systemic_sclerosis",
        "methods": list(report.METHODS),
        "configurations": list(report.CONFIGURATIONS),
        "arms": list(report.ARMS),
        "training": "none; completed immutable out-of-fold predictions only",
    }
    (root / "pooling_manifest.json").write_text(
        json.dumps({"protocol": protocol, "output_sha256": {csv.name: digest}, "output_sha256_excludes": ["pooling_manifest.json"]}),
        encoding="utf-8",
    )


def test_manifest_hash_failure_and_atomic_report(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    frame = patient_table()
    primary = (frame.configuration == "full") & (frame.method == "primary_mean")
    # Derive the acceptance table from the file: 10 shared correct, 3 shared
    # wrong, and one NOVAE-only correct patient.
    nmf_wrong = primary & (frame.arm == "nmf") & frame.patient.isin(["patient-11", "patient-12", "patient-13", "patient-14"])
    novae_wrong = primary & (frame.arm == "novae") & frame.patient.isin(["patient-11", "patient-12", "patient-13"])
    frame.loc[nmf_wrong, "predicted_label"] = frame.loc[nmf_wrong, "true_label"].map({"healthy": "systemic_sclerosis", "systemic_sclerosis": "healthy"})
    frame.loc[novae_wrong, "predicted_label"] = frame.loc[novae_wrong, "true_label"].map({"healthy": "systemic_sclerosis", "systemic_sclerosis": "healthy"})
    frame.loc[primary, "correct"] = frame.loc[primary, "predicted_label"].eq(frame.loc[primary, "true_label"])
    _write_input(input_root, frame)
    inventory = report.validate_manifest_inventory(input_root / "pooling_manifest.json", input_root)
    assert inventory["patient_predictions_long.csv"] == hashlib.sha256((input_root / "patient_predictions_long.csv").read_bytes()).hexdigest()
    output = tmp_path / "report"
    manifest = report.run_mcnemar(input_root=input_root, output_root=output)
    assert output.is_dir()
    assert len(pd.read_csv(output / "exact_mcnemar_all.csv")) == 12
    primary_result = json.loads((output / "primary_exact_mcnemar.json").read_text(encoding="utf-8"))["result"]
    assert {key: primary_result[key] for key in ("both_correct", "both_wrong", "nmf_only_correct", "novae_only_correct", "discordant", "n")} == {
        "both_correct": 10,
        "both_wrong": 3,
        "nmf_only_correct": 0,
        "novae_only_correct": 1,
        "discordant": 1,
        "n": 14,
    }
    assert primary_result["exact_two_sided_p"] == pytest.approx(1.0)
    assert manifest["source_unchanged_before_after"] is True
    with pytest.raises(report.ContractError, match="no overwrite"):
        report.run_mcnemar(input_root=input_root, output_root=output)
    (input_root / "patient_predictions_long.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(report.ContractError, match="hash mismatch"):
        report.validate_manifest_inventory(input_root / "pooling_manifest.json", input_root)


def test_cli_bootstrap_from_non_repo_cwd(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_novae_patient_level_mcnemar.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--input-root" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX Bash")
def test_launcher_render_contract(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    run_root = tmp_path / "run"
    env = {
        **os.environ,
        "NOVAE_PATIENT_LEVEL_MCNEMAR_REPO_DIR": str(root),
        "NOVAE_PATIENT_LEVEL_MCNEMAR_INPUT_ROOT": str(tmp_path / "immutable_input"),
        "NOVAE_PATIENT_LEVEL_MCNEMAR_RUN_ROOT": str(run_root),
        "NOVAE_PATIENT_LEVEL_MCNEMAR_OUTPUT_ROOT": str(run_root / "output"),
        "NOVAE_PATIENT_LEVEL_MCNEMAR_JOB_SCRIPT": str(run_root / "report.sbatch"),
        "NOVAE_PATIENT_LEVEL_MCNEMAR_LOG_DIR": str(run_root / "logs"),
    }
    subprocess.run(["bash", str(root / "scripts/submit_novae_patient_level_mcnemar.sh"), "--render-only"], env=env, check=True, capture_output=True, text=True)
    job = (run_root / "report.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=1" in job
    assert "#SBATCH --mem=16gb" in job
    assert 'CUDA_VISIBLE_DEVICES=""' in job
    assert "run_novae_patient_level_mcnemar.py" in job
