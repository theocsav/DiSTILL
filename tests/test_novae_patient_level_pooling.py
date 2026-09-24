"""Synthetic contract tests for predeclared patient-level prediction pooling."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_novae_patient_level_pooling as pooling


def synthetic_predictions() -> pd.DataFrame:
    rows = []
    # Four healthy and ten systemic-sclerosis patients, with 61/103 FOVs.
    counts = [16, 15, 15, 15] + [10] * 9 + [13]
    labels = ["healthy"] * 4 + ["systemic_sclerosis"] * 10
    item = 0
    for fold, (count, label) in enumerate(zip(counts, labels, strict=True), 1):
        for _position in range(count):
            probability = 0.75 if label == "systemic_sclerosis" else 0.25
            rows.append({"item_id": f"fov-{item:03d}", "outer_fold": fold, "test_group": f"patient-{fold:02d}", "true_label": label, "predicted_label": "systemic_sclerosis" if probability >= 0.5 else "healthy", "decision_threshold": 0.5, "positive_class": "systemic_sclerosis", "positive_class_probability": probability})
            item += 1
    return pd.DataFrame(rows)


def test_direct_cli_bootstrap_help_from_repo_and_nonrepo_cwd(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    script = root / "scripts" / "run_novae_patient_level_pooling.py"
    for cwd in (root, tmp_path):
        result = subprocess.run([sys.executable, str(script), "--help"], cwd=cwd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "--ablation-root" in result.stdout


def test_pooling_math_and_exact_tie_fallback() -> None:
    frame = synthetic_predictions()
    # Patient one has an exact vote tie and a mean below threshold.
    frame.loc[:15, "positive_class_probability"] = [0.51] * 8 + [0.0] * 8
    frame.loc[:15, "predicted_label"] = ["systemic_sclerosis"] * 8 + ["healthy"] * 8
    pooled = pooling.pool_patient_predictions(pooling._normalise_prediction(frame, Path("synthetic")), "full", "nmf")
    row = pooled[(pooled.patient == "patient-01") & (pooled.method == "sensitivity_majority_vote")].iloc[0]
    assert bool(row.tie) is True
    assert row.score == pytest.approx(0.5)
    assert row.predicted_label == "healthy"
    assert pooled[pooled.method == "primary_mean"].shape[0] == 14


def test_alignment_and_contract_failures() -> None:
    frame = pooling._normalise_prediction(synthetic_predictions(), Path("synthetic"))
    frames = {(configuration, arm): frame.copy() for configuration in pooling.CONFIGURATIONS for arm in pooling.ARMS}
    pooling.validate_prediction_alignment(frames)
    frames["composition_only", "novae"].loc[0, "item_id"] = "wrong"
    with pytest.raises(pooling.ContractError, match="alignment"):
        pooling.validate_prediction_alignment(frames)
    bad = synthetic_predictions()
    bad.loc[0, "positive_class_probability"] = 2
    with pytest.raises(pooling.ContractError, match=r"\[0,1\]"):
        pooling._normalise_prediction(bad, Path("synthetic"))
    bad = synthetic_predictions()
    bad.loc[0, "predicted_label"] = "systemic_sclerosis"
    with pytest.raises(pooling.ContractError, match="inconsistent"):
        pooling._normalise_prediction(bad, Path("synthetic"))
    bad = synthetic_predictions()
    bad.loc[0, "decision_threshold"] = 0.4
    with pytest.raises(pooling.ContractError, match="threshold"):
        pooling._normalise_prediction(bad, Path("synthetic"))
    bad = synthetic_predictions()
    bad.loc[0, "test_group"] = "patient-02"
    with pytest.raises(pooling.ContractError, match="one outer fold|one true class"):
        pooling._normalise_prediction(bad, Path("synthetic"))
    bad = synthetic_predictions()
    bad.loc[1, "item_id"] = bad.loc[0, "item_id"]
    with pytest.raises(pooling.ContractError, match="duplicate"):
        pooling._normalise_prediction(bad, Path("synthetic"))


def test_metric_recompute_and_no_overwrite(tmp_path: Path) -> None:
    frame = pooling._normalise_prediction(synthetic_predictions(), Path("synthetic"))
    rows = pd.concat([pooling.pool_patient_predictions(frame, configuration, arm) for configuration in pooling.CONFIGURATIONS for arm in pooling.ARMS], ignore_index=True)
    metrics, matrices = pooling.patient_metrics(rows)
    assert len(metrics) == 4 * 2 * 3 * 4
    assert len(matrices) == 4 * 2 * 3
    assert set(metrics.metric) == {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"}
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(pooling.ContractError, match="overwrite"):
        pooling.run_pooling(full_primary_dir=tmp_path / "full", ablation_root=tmp_path / "ablation", output_root=out)


def _write_manifest(root: Path, name: str, entries: dict[str, str], **extra: object) -> None:
    payload = {"output_sha256": entries, "output_sha256_excludes": [name], **extra}
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


def _relationship_manifests(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    full = tmp_path / "full"
    ablation = tmp_path / "ablation"
    full.mkdir()
    aggregate_dir = ablation / "aggregate"
    aggregate_dir.mkdir(parents=True)
    inputs = {"synthetic": True}
    primary_code = pooling.ablation._primary_code_hashes()
    full_manifest = full / "run_manifest.json"
    full_manifest.write_text(json.dumps({"protocol": pooling.ablation.primary.PROTOCOL, "code_sha256": primary_code, "input_sha256": inputs}), encoding="utf-8")
    code = {"evaluator": pooling.sha256(pooling.ablation.EVALUATOR), "orchestrator": pooling.sha256(Path(pooling.ablation.__file__)), "launcher": pooling.sha256(pooling.ablation.LAUNCHER)}
    config_manifests = []
    for configuration in pooling.CONFIGURATIONS[1:]:
        directory = ablation / configuration
        directory.mkdir()
        manifest = directory / "config_manifest.json"
        manifest.write_text(json.dumps({"ablation": configuration, "protocol": pooling.ablation.protocol_for(configuration), "inputs": inputs, "environment": {"nmf": pooling.ablation.expected_environment(configuration, "nmf_prop_"), "novae": pooling.ablation.expected_environment(configuration, "novae_prop_")}, "code_sha256": code}), encoding="utf-8")
        config_manifests.append(manifest)
    aggregate = aggregate_dir / "run_manifest.json"
    aggregate.write_text(json.dumps({"protocol": {configuration: pooling.ablation.protocol_for(configuration) for configuration in pooling.CONFIGURATIONS}, "completed_ablations": list(pooling.CONFIGURATIONS), "full_primary_output": str(full), "full_primary_manifest_sha256": pooling.sha256(full_manifest), "full_primary_unchanged": True, "inputs": inputs, "code_sha256": code}), encoding="utf-8")
    return full, ablation, [full_manifest, aggregate, *config_manifests]


def test_manifest_relationship_mutation_is_rejected(tmp_path: Path) -> None:
    full, ablation, manifests = _relationship_manifests(tmp_path)
    pooling._validate_source_relationships(full, ablation, manifests)
    config = manifests[2]
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["environment"]["nmf"]["NICHERUNNER_TOP_ENRICHMENT_FEATURES"] = "999"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(pooling.ContractError, match="environment"):
        pooling._validate_source_relationships(full, ablation, manifests)


def test_manifest_traversal_and_hash_failures(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    output = root / "fold_predictions.csv"
    output.write_text("x\n1\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = root / "run_manifest.json"
    _write_manifest(root, manifest.name, {output.name: digest})
    assert pooling.validate_manifest_inventory(manifest, root) == {output.name: digest}
    output.write_text("x\n2\n", encoding="utf-8")
    with pytest.raises(pooling.ContractError, match="hash mismatch"):
        pooling.validate_manifest_inventory(manifest, root)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    traversal = root / "traversal.json"
    _write_manifest(root, traversal.name, {"../outside.txt": hashlib.sha256(outside.read_bytes()).hexdigest()})
    with pytest.raises(pooling.ContractError, match="escapes"):
        pooling.validate_manifest_inventory(traversal, root)


def test_output_overlap_is_rejected_in_both_directions(tmp_path: Path) -> None:
    full = tmp_path / "full"
    ablation = tmp_path / "ablation"
    full.mkdir()
    ablation.mkdir()
    with pytest.raises(pooling.ContractError, match="overlaps"):
        pooling.run_pooling(full_primary_dir=full, ablation_root=ablation, output_root=full / "nested")
    output = tmp_path / "output"
    with pytest.raises(pooling.ContractError, match="overlaps"):
        pooling.run_pooling(full_primary_dir=output / "nested-source", ablation_root=ablation, output_root=output)


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX Bash")
def test_launcher_render_resources_and_gpu_off(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    run_root = tmp_path / "run"
    env = {**os.environ, "NOVAE_PATIENT_LEVEL_POOLING_REPO_DIR": str(root), "NOVAE_PATIENT_LEVEL_POOLING_FULL_PRIMARY_DIR": str(tmp_path / "full"), "NOVAE_PATIENT_LEVEL_POOLING_ABLATION_ROOT": str(tmp_path / "ablation"), "NOVAE_PATIENT_LEVEL_POOLING_RUN_ROOT": str(run_root), "NOVAE_PATIENT_LEVEL_POOLING_OUTPUT_ROOT": str(run_root / "output"), "NOVAE_PATIENT_LEVEL_POOLING_JOB_SCRIPT": str(run_root / "pool.sbatch"), "NOVAE_PATIENT_LEVEL_POOLING_LOG_DIR": str(run_root / "logs")}
    subprocess.run(["bash", str(root / "scripts/submit_novae_patient_level_pooling.sh"), "--render-only"], env=env, check=True, capture_output=True, text=True)
    job = (run_root / "pool.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=1" in job
    assert "#SBATCH --mem=16gb" in job
    assert 'CUDA_VISIBLE_DEVICES=""' in job
    assert "run_novae_patient_level_pooling.py" in job
