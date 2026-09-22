from __future__ import annotations

from types import SimpleNamespace
import os
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.build_novae_fov_features import (
    ContractError,
    DOMAIN_KEY,
    VALIDITY_KEY,
    _assert_finite_matrix,
    _domain_assignments,
    _validate_fov_consistency,
    _validate_provenance,
    composition_features,
    enrichment_features,
    natural_domain_order,
    niche_gene_features,
)


def _assignments():
    return pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2"],
            "Disease_State": ["healthy", "healthy", "SSc", "SSc"],
            "fov_key": ["p1_a", "p1_a", "p2_a", "p2_b"],
            "_domain": ["L0", "L1", "L0", pd.NA],
            "_valid": [True, True, True, False],
            "_assigned": [True, True, True, False],
            "CenterX_global_px": [0.0, 1.0, 0.0, 0.0],
            "CenterY_global_px": [0.0, 0.0, 0.0, 0.0],
            "Area": [225.0, 225.0, 225.0, 225.0],
        },
        index=["c0", "c1", "c2", "c3"],
    )


def test_composition_preserves_zero_assigned_fov_and_sums():
    counts, props = composition_features(_assignments(), pd.Index(["p1_a", "p2_a", "p2_b"], name="fov_key"), ["L0", "L1"])
    assert counts.loc["p2_b"].tolist() == [0, 0]
    assert props.loc["p2_b"].tolist() == [0.0, 0.0]
    assert props.loc["p1_a"].sum() == pytest.approx(1.0)


def test_enrichment_does_not_leak_between_fovs_and_singleton_is_zero():
    result = enrichment_features(_assignments(), pd.Index(["p1_a", "p2_a", "p2_b"], name="fov_key"), ["L0", "L1"])
    assert np.isfinite(result.to_numpy()).all()
    assert result.loc["p2_b"].eq(0).all()
    assert result.loc["p1_a", "enrichment_1-1"] == pytest.approx(-1.0)
    assert result.loc["p1_a", "enrichment_1-2"] == pytest.approx(np.log2(1.5))
    # p1_a has one L0/L1 pair; adding p2_a cannot alter it.
    other = _assignments().copy()
    other.loc["c2", "CenterX_global_px"] = 0.0
    assert result.loc["p1_a"].equals(enrichment_features(other, pd.Index(["p1_a", "p2_a", "p2_b"]), ["L0", "L1"]).loc["p1_a"])


def test_backed_sparse_matrix_finite_and_nonfinite():
    class FakeBackedMatrix:
        def __init__(self, values):
            self.values = sparse.csr_matrix(values)
            self.shape = self.values.shape

        def to_memory(self):
            return self.values

        def __getitem__(self, index):
            return self.values[index]

    _assert_finite_matrix(FakeBackedMatrix([[1.0, 0.0], [0.0, 2.0]]))
    with pytest.raises(ContractError, match="non-finite"):
        _assert_finite_matrix(FakeBackedMatrix([[1.0, np.nan]]))


def test_nonfinite_assigned_spatial_and_expression_fail_closed():
    assignments = _assignments()
    assignments.loc["c0", "Area"] = np.nan
    with pytest.raises(ContractError, match="finite"):
        enrichment_features(assignments, pd.Index(["p1_a", "p2_a", "p2_b"]), ["L0", "L1"])
    assignments = _assignments()
    dense = SimpleNamespace(X=np.array([[1, np.nan], [3, 4], [5, 7], [100, 100]], dtype=float), var_names=np.array(["g1", "g2"]), raw=None)
    with pytest.raises(ContractError, match="non-finite"):
        niche_gene_features(assignments, dense, pd.Index(["p1_a", "p2_a", "p2_b"]), ["L0", "L1"])


def test_niche_gene_dense_and_sparse_are_equal_and_absent_is_zero():
    assignments = _assignments()
    dense = SimpleNamespace(X=np.array([[1, 2], [3, 4], [5, 7], [100, 100]], dtype=float), var_names=np.array(["g1", "g2"]), raw=None)
    sparse_adata = SimpleNamespace(X=sparse.csr_matrix(dense.X), var_names=dense.var_names, raw=None)
    index = pd.Index(["p1_a", "p2_a", "p2_b"], name="fov_key")
    d = niche_gene_features(assignments, dense, index, ["L0", "L1"])
    s = niche_gene_features(assignments, sparse_adata, index, ["L0", "L1"])
    pd.testing.assert_frame_equal(d, s)
    assert d.loc["p2_b"].eq(0).all()


def test_domain_order_and_invalid_vocabulary():
    assert natural_domain_order(["L2", "L0", "L1"]) == ["L0", "L1", "L2"]
    with pytest.raises(ContractError):
        natural_domain_order(["L0", "domain_1"])
    with pytest.raises(ContractError):
        natural_domain_order(["L0", "L9"])


def test_assignment_contract_rejects_valid_missing_and_invalid_labeled():
    base = pd.DataFrame(
        {"unique_cell_id": ["a", "b"], "patient": ["p", "p"], "Disease_State": ["H", "H"], "fov": ["x", "x"]},
        index=["obs_a", "obs_b"],
    )
    novae = base.drop(columns=["fov"]).copy()
    novae[DOMAIN_KEY] = ["L0", pd.NA]
    novae[VALIDITY_KEY] = [True, True]
    with pytest.raises(ContractError):
        _domain_assignments(base, novae)

    novae[VALIDITY_KEY] = [True, False]
    assigned = _domain_assignments(base, novae)
    assert assigned.loc["a", "_assigned"] and not assigned.loc["b", "_assigned"]


def test_mixed_fov_metadata_is_rejected_before_aggregation():
    metadata = pd.DataFrame({"fov_key": ["f1", "f1"], "patient": ["p1", "p2"], "Disease_State": ["H", "H"]})
    with pytest.raises(ContractError, match="mixed"):
        _validate_fov_consistency(metadata, "synthetic")
    metadata.loc[1, "patient"] = "p1"
    metadata.loc[1, "Disease_State"] = "SSc"
    with pytest.raises(ContractError, match="mixed"):
        _validate_fov_consistency(metadata, "synthetic")


@pytest.mark.skipif(os.name == "nt", reason="launcher rendering requires a POSIX shell/filesystem")
def test_launcher_renders_cpu_contract(tmp_path):
    root = tmp_path / "run"
    env = {
        "NOVAE_REPO_DIR": str(tmp_path / "repo"),
        "NOVAE_FOV_BASE_H5AD": str(tmp_path / "base.h5ad"),
        "NOVAE_FOV_NOVAE_H5AD": str(tmp_path / "novae.h5ad"),
        "NOVAE_FOV_FEATURE_DIR": str(tmp_path / "features"),
        "NOVAE_FOV_SOURCE_OUTPUT_DIR": str(tmp_path / "source"),
        "NOVAE_FOV_RUN_ROOT": str(root),
        "NOVAE_FOV_OUTPUT_DIR": str(root / "features"),
        "NOVAE_FOV_LOG_DIR": str(root / "logs"),
        "NOVAE_FOV_JOB_SCRIPT": str(root / "job.sbatch"),
        "NOVAE_FOV_CONDA_ENV": str(tmp_path / "env"),
    }
    command = ["bash", "scripts/submit_novae_fov_features.sh", "--render-only"]
    result = subprocess.run(command, env={**__import__("os").environ, **env}, capture_output=True, text=True, check=True)
    assert "Rendered sbatch script" in result.stdout
    rendered = (root / "job.sbatch").read_text()
    assert "--cpus-per-task=4" in rendered
    assert "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONHASHSEED=42" in rendered
    assert 'export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="void"' in rendered
    assert "--expected-domains L0,L1,L2,L3,L4,L5,L6,L7,L8" in rendered


def test_unknown_validity_and_provenance_mismatch_fail_closed():
    base = pd.DataFrame({"unique_cell_id": ["a", "b"], "patient": ["p", "p"], "Disease_State": ["H", "H"], "fov": ["x", "x"]})
    novae = base.drop(columns=["fov"]).copy()
    novae[DOMAIN_KEY] = ["L0", pd.NA]
    novae[VALIDITY_KEY] = ["maybe", False]
    with pytest.raises(ContractError, match="unrecognized"):
        _domain_assignments(base, novae)
    provenance = {
        "analysis_scope": "exploratory", "reference": "all", "inference_mode": "zero_shot",
        "dataset_id": "skin_visium_ssc_paired_cpu_calibrated", "coordinate_strategy": "visium_explicit_scale",
        "primary_resolution": 1.0, "domain_key": DOMAIN_KEY, "neighborhood_valid_key": VALIDITY_KEY,
        "accelerator": "cpu", "device": "cpu", "workers": 0,
        "confirmatory_held_out_classification_allowed": False,
        "input_sha256": "262418e8e7ed06de805e940406f3ae9e41487ce085da1ae8f940c81f95daf6dd",
        "checkpoint_sha256": "1422f9f72d6e532921bf8a90f0996f1c46c6891f6ecbc73e404521ec5aa7b04a",
        "minimum_domain_assignment_coverage": 0.70,
        "deterministic_policy": {"requested": True, "effective": True}, "seed": 42,
    }
    round_tripped = {**provenance, "deterministic_policy": {"requested": np.bool_(True), "effective": np.bool_(True)}}
    assert _validate_provenance(SimpleNamespace(uns={"novae_pilot_provenance": round_tripped})) == round_tripped
    assert _validate_provenance(SimpleNamespace(uns={"novae_pilot_provenance": provenance})) == provenance
    for key, value in (("input_sha256", "bad"), ("checkpoint_sha256", "bad"), ("minimum_domain_assignment_coverage", 0.8)):
        broken = {**provenance, key: value}
        with pytest.raises(ContractError, match="provenance"):
            _validate_provenance(SimpleNamespace(uns={"novae_pilot_provenance": broken}))
    broken = {**provenance, "deterministic_policy": {"requested": True, "effective": False}}
    with pytest.raises(ContractError, match="deterministic"):
        _validate_provenance(SimpleNamespace(uns={"novae_pilot_provenance": broken}))
