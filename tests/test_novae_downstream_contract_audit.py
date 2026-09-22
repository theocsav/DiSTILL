"""Synthetic-only tests for the NOVAE downstream contract evidence gate."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

import scripts.audit_novae_downstream_contract as audit_module
from scripts.audit_novae_downstream_contract import Candidate, ContractAuditError, audit_candidate, run_audit


_PARQUET_ENGINE = bool(importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"))
_PARQUET_TABLES: dict[str, pd.DataFrame] = {}


@pytest.fixture(autouse=True)
def _synthetic_table_loader(monkeypatch):
    original = audit_module._load_table

    def load(path):
        if str(path) in _PARQUET_TABLES:
            return _PARQUET_TABLES[str(path)].copy()
        return original(path)

    monkeypatch.setattr(audit_module, "_load_table", load)
    _PARQUET_TABLES.clear()
    yield
    _PARQUET_TABLES.clear()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    if _PARQUET_ENGINE:
        frame.to_parquet(path)
    else:
        path.write_text("synthetic parquet placeholder", encoding="utf-8")
    _PARQUET_TABLES[str(path)] = frame.copy()


def _read_parquet(path: Path) -> pd.DataFrame:
    if str(path) in _PARQUET_TABLES:
        return _PARQUET_TABLES[str(path)].copy()
    return pd.read_parquet(path)


def _fixture(
    root: Path,
    *,
    disease_key="Disease_State",
    misalign=False,
    conflict=False,
    patient_conflict=False,
    disease_conflict=False,
    extra_composition=False,
    omit_nmf=False,
    raw_reverse=False,
    raw_extra=False,
    canonical_subset=False,
    target_reverse=False,
    group_reverse=False,
    composition=None,
):
    source_path = root / "source.h5ad"
    run = root / "run"
    (run / "MLP_FOVFeatures_inputs").mkdir(parents=True)
    ids = ["P1_A_1", "P1_A_2", "P2_B_1", "P2_B_2"]
    obs = pd.DataFrame(
        {
            "patient": ["P1", "P1", "P2", "P2"],
            "fov": ["A", "A", "B", "B"],
            disease_key: ["healthy", "healthy", "case", "case"],
            "unique_cell_id": ids,
        },
        index=ids,
    )
    ad.AnnData(X=np.ones((4, 2)), obs=obs).write_h5ad(source_path)
    nmf_obs = obs.copy()
    nmf_obs["NMF_factor"] = ["0", "1", "0", "1"]
    if not omit_nmf:
        nmf_obs["dominant_nmf_factor"] = ["0", "1", "0", "1"]
    if patient_conflict:
        nmf_obs.iloc[0, nmf_obs.columns.get_loc("patient")] = "P9"
        nmf_obs["field_of_view"] = ["P1_A", "P1_A", "P2_B", "P2_B"]
    if disease_conflict:
        nmf_obs.iloc[0, nmf_obs.columns.get_loc(disease_key)] = "case"
        nmf_obs["field_of_view"] = ["P1_A", "P1_A", "P2_B", "P2_B"]
    if misalign:
        nmf_obs = nmf_obs.iloc[[1, 0, 2, 3]]
    ad.AnnData(X=np.ones((4, 2)), obs=nmf_obs).write_h5ad(run / "cosmx_with_nmf.h5ad")

    post = obs.reset_index(drop=True)
    post["NMF_factor"] = ["0", "1", "0", "1"]
    if conflict:
        post.loc[1, disease_key] = "case"
    post.to_csv(run / "post_nmf_obs.csv", index=False)

    all_fovs = ["P1_A", "P2_B"]
    canonical = ["P1_A"] if canonical_subset else all_fovs
    raw_index = list(reversed(all_fovs)) if raw_reverse else list(all_fovs)
    if raw_extra:
        raw_index.append("P3_C")
    raw_values = np.arange(len(raw_index), dtype=float)
    pd.DataFrame({"enrichment_0": raw_values}, index=pd.Index(raw_index, name="field_of_view")).to_csv(
        run / "enrichment_features_fov.csv"
    )
    _write_parquet(
        pd.DataFrame({"niche_0": raw_values + 1}, index=pd.Index(raw_index, name="field_of_view")),
        run / "niche_gene_features_fov.parquet",
    )

    props = composition or {"P1_A": (0.5, 0.5), "P2_B": (0.5, 0.5)}
    combined_data = {
        "nmf_prop_0": [props[fov][0] for fov in canonical],
        "nmf_prop_1": [props[fov][1] for fov in canonical],
        "enrichment_0": np.arange(len(canonical), dtype=float),
    }
    if extra_composition:
        combined_data["nmf_prop_unexpected"] = [0.0 for _ in canonical]
    combined = pd.DataFrame(
        combined_data,
        index=pd.Index(canonical, name="field_of_view"),
    )
    _write_parquet(combined, run / "MLP_FOVFeatures_inputs" / "combined_features_filtered.parquet")
    targets = pd.DataFrame({"Disease_State": ["healthy" if fov == "P1_A" else "case" for fov in canonical]}, index=combined.index)
    groups = pd.DataFrame({"patient": [fov.split("_")[0] for fov in canonical]}, index=combined.index)
    if target_reverse:
        targets = targets.iloc[::-1]
    if group_reverse:
        groups = groups.iloc[::-1]
    _write_parquet(targets, run / "MLP_FOVFeatures_inputs" / "targets_y.parquet")
    _write_parquet(groups, run / "MLP_FOVFeatures_inputs" / "groups.parquet")
    (run / "post_nmf_artifacts.json").write_text("{}", encoding="utf-8")
    return Candidate("historical_164", source_path, run)


def _failed(result, name):
    return any(row["check"] == name and not row["passed"] for row in result["checks"])


def test_complete_contract_is_eligible_and_reconciles_half_props(tmp_path):
    result = audit_candidate(_fixture(tmp_path))
    assert result["accepted"]
    assert result["composition_proportions"]["P1_A"] == {"nmf_prop_0": 0.5, "nmf_prop_1": 0.5}
    assert result["counts"]["canonical_fov_rows"] == 2


def test_raw_reorder_and_extra_rows_are_allowed(tmp_path):
    result = audit_candidate(_fixture(tmp_path, raw_reverse=True, raw_extra=True))
    assert result["accepted"]
    enrichment = next(row for row in result["index_comparisons"] if row["artifact"] == "enrichment")
    assert enrichment["extra"] == ["P3_C"] and not enrichment["order_equal"]


def test_canonical_target_or_group_reorder_is_rejected(tmp_path):
    assert not audit_candidate(_fixture(tmp_path / "target", target_reverse=True))["accepted"]
    assert not audit_candidate(_fixture(tmp_path / "group", group_reverse=True))["accepted"]


def test_canonical_subset_is_accepted_and_exclusions_are_recorded(tmp_path):
    result = audit_candidate(_fixture(tmp_path, canonical_subset=True))
    assert result["accepted"]
    assert result["excluded_post_fov_ids"] == ["P2_B"]
    assert result["counts"]["excluded_post_fov_rows"] == 1


@pytest.mark.parametrize(
    "kwargs,check",
    [
        ({"composition": {"P1_A": (0.4, 0.6), "P2_B": (0.5, 0.5)}}, "combined_composition_reconciles"),
        ({"composition": {"P1_A": (-0.1, 1.1), "P2_B": (0.5, 0.5)}}, "combined_composition_nonnegative"),
        ({"composition": {"P1_A": (0.3, 0.3), "P2_B": (0.5, 0.5)}}, "combined_composition_normalized"),
    ],
)
def test_invalid_composition_props_are_rejected(tmp_path, kwargs, check):
    result = audit_candidate(_fixture(tmp_path, **kwargs))
    assert not result["accepted"] and _failed(result, check)


def test_unexpected_composition_column_is_rejected(tmp_path):
    result = audit_candidate(_fixture(tmp_path, extra_composition=True))
    assert not result["accepted"] and _failed(result, "combined_no_unexpected_composition_columns")


def test_missing_composition_column_is_rejected(tmp_path):
    candidate = _fixture(tmp_path)
    path = candidate.run_output / "MLP_FOVFeatures_inputs" / "combined_features_filtered.parquet"
    frame = _read_parquet(path).drop(columns=["nmf_prop_1"])
    _write_parquet(frame, path)
    result = audit_candidate(candidate)
    assert not result["accepted"] and _failed(result, "combined_composition_columns")


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, "not-a-number"])
def test_bad_raw_features_are_rejected(tmp_path, bad_value):
    candidate = _fixture(tmp_path)
    path = candidate.run_output / "enrichment_features_fov.csv"
    frame = pd.read_csv(path, index_col=0)
    if isinstance(bad_value, str):
        frame = frame.astype(object)
    frame.iloc[0, 0] = bad_value
    frame.to_csv(path)
    result = audit_candidate(candidate)
    assert not result["accepted"] and _failed(result, "enrichment_finite_numeric")


def test_duplicate_raw_feature_columns_are_rejected(tmp_path, monkeypatch):
    candidate = _fixture(tmp_path)
    original = audit_module._load_table

    def duplicate_loader(path):
        frame = original(path)
        if path.name == "enrichment_features_fov.csv":
            frame = pd.concat([frame, frame.iloc[:, [0]]], axis=1)
            frame.columns = ["enrichment_0", "enrichment_0"]
        return frame

    monkeypatch.setattr(audit_module, "_load_table", duplicate_loader)
    result = audit_candidate(candidate)
    assert not result["accepted"] and _failed(result, "enrichment_columns")


def test_missing_artifact_is_explicit_and_rejected(tmp_path):
    candidate = _fixture(tmp_path)
    (candidate.run_output / "MLP_FOVFeatures_inputs" / "groups.parquet").unlink()
    result = audit_candidate(candidate)
    assert not result["artifacts"]["groups"]["present"] and not result["accepted"]


def test_source_nmf_patient_value_conflict_is_rejected(tmp_path):
    result = audit_candidate(_fixture(tmp_path, patient_conflict=True))
    assert not result["accepted"] and _failed(result, "source_nmf_patient_values")


def test_source_nmf_disease_value_conflict_is_rejected(tmp_path):
    result = audit_candidate(_fixture(tmp_path, disease_conflict=True))
    assert not result["accepted"] and _failed(result, "source_nmf_disease_state_values")


def test_h5ad_row_misalignment_rejected(tmp_path):
    result = audit_candidate(_fixture(tmp_path, misalign=True))
    assert not result["accepted"] and _failed(result, "source_nmf_obs_order")


def test_label_conflict_is_rejected(tmp_path):
    result = audit_candidate(_fixture(tmp_path, conflict=True))
    assert not result["accepted"] and _failed(result, "fov_one_patient_one_label")


def test_lowercase_disease_state_is_recorded_compatibility(tmp_path):
    result = audit_candidate(_fixture(tmp_path, disease_key="disease_state"))
    assert result["accepted"]
    assert result["resolved_keys"]["source"]["disease_key"] == "disease_state"


def test_missing_nmf_column_is_rejected(tmp_path):
    result = audit_candidate(_fixture(tmp_path, omit_nmf=True))
    assert not result["accepted"] and any("dominant_nmf_factor" in row["detail"] for row in result["checks"] if not row["passed"])


def test_read_only_and_atomic_publication(tmp_path):
    candidate = _fixture(tmp_path)
    digest_before = hashlib.sha256(candidate.source_h5ad.read_bytes()).hexdigest()
    output = tmp_path / "audit"
    run_audit((candidate,), output)
    assert (output / "novae_downstream_contract_audit.json").exists()
    assert hashlib.sha256(candidate.source_h5ad.read_bytes()).hexdigest() == digest_before
    with pytest.raises(ContractAuditError):
        run_audit((candidate,), output)
    assert not list(tmp_path.glob(".audit.*.partial"))


def test_launcher_render_and_injection_rejection(tmp_path):
    root = Path(__file__).parents[1]
    script = root / "scripts" / "submit_novae_downstream_contract_audit.sh"
    job = tmp_path / "job.sbatch"
    env = {**os.environ, "NOVAE_REPO_DIR": str(root), "NOVAE_CONTRACT_AUDIT_RUN_ROOT": str(tmp_path / "run"), "NOVAE_CONTRACT_AUDIT_JOB_SCRIPT": str(job)}
    completed = subprocess.run([str(script), "--render-only"], env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    rendered = job.read_text()
    assert "--cpus-per-task=1" in rendered and "--mem=64gb" in rendered
    assert "gpu" not in rendered.lower() and "audit_novae_downstream_contract.py" in rendered
    rejected = subprocess.run([str(script), "--render-only"], env={**env, "NOVAE_ACCOUNT": "safe;rm"}, text=True, capture_output=True)
    assert rejected.returncode != 0
