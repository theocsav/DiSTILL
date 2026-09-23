from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import run_novae_spatial_biological_validation as validation


class FakeAnnData:
    def __init__(self, *, omit_post_patient: bool = False, omit_l8: bool = False) -> None:
        n = 19
        ids = [f"cell_{i}" for i in range(n)]
        domains = [f"L{i}" for i in range(9)] + [f"L{i % 8}" for i in range(9)] + [""]
        if omit_l8:
            domains[8] = "L0"
        self.obs_names = pd.Index(ids)
        self.obs = pd.DataFrame({"unique_cell_id": ids, "sample_id": ["S1"] * 10 + ["S2"] * 9, "patient": ["P1"] * 10 + ["P2"] * 9, "Disease_State": ["healthy"] * 10 + ["disease"] * 9, validation.DOMAIN_KEY: domains, validation.VALID_KEY: [True] * 18 + [False]}, index=ids)
        self.n_obs = n
        self.obsm = {"spatial": np.arange(n * 2, dtype=float).reshape(n, 2)}
        self.obsp = {validation.GRAPH_KEY: sparse.block_diag((sparse.diags([np.ones(9), np.ones(9)], [-1, 1], shape=(10, 10)), sparse.diags([np.ones(8), np.ones(8)], [-1, 1], shape=(9, 9))), format="csr")}
        self.X = np.arange(1, n * 3 + 1, dtype=float).reshape(n, 3)
        self.var_names = pd.Index(["g0", "g1", "g2"])
        self.uns = {"novae_pilot_provenance": {"analysis_scope": "exploratory", "reference": "all", "dataset_id": "skin_visium_ssc_paired_cpu_calibrated", "coordinate_strategy": "visium_explicit_scale", "domain_key": validation.DOMAIN_KEY, "neighborhood_valid_key": validation.VALID_KEY, "primary_resolution": 1.0, "accelerator": "cpu", "device": "cpu", "workers": 0, "seed": 42, "input_sha256": validation.EXPECTED_NOVAE_INPUT_SHA256, "checkpoint_sha256": validation.EXPECTED_NOVAE_CHECKPOINT_SHA256, "deterministic_policy": {"requested": True, "effective": True}}}
        self.omit_post_patient = omit_post_patient

    def post_frame(self) -> pd.DataFrame:
        ids = self.obs_names.tolist()
        frame = pd.DataFrame({"unique_cell_id": ids, "NMF_factor": [i % 9 for i in range(len(ids))], "patient": self.obs["patient"].to_numpy(), "Disease_State": self.obs["Disease_State"].to_numpy(), "sample_id": self.obs["sample_id"].to_numpy()})
        if self.omit_post_patient:
            frame = frame.drop(columns=["patient"])
        return frame


def chain_graph(n: int = 4) -> sparse.csr_matrix:
    return sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="csr")


def test_graph_rejects_cross_slide_and_asymmetry() -> None:
    graph = chain_graph()
    with pytest.raises(validation.ContractError, match="cross-slide"):
        validation.validate_graph(graph, ["A", "A", "B", "B"], 4)
    with pytest.raises(validation.ContractError, match="symmetric"):
        validation.validate_graph(sparse.csr_matrix(np.array([[0, 1], [0, 0.0]])), ["A", "A"], 2)
    with pytest.raises(validation.ContractError, match="diagonal"):
        validation.validate_graph(sparse.csr_matrix(np.array([[1, 0], [0, 0.0]])), ["A", "A"], 2)


def test_pas_ties_and_zero_degree_are_explicit() -> None:
    graph = sparse.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float))
    result = validation.pas(["A", "A", "B"], graph)
    assert result["tie_spots"] == 1
    assert result["zero_degree_spots"] == 0
    tie = validation.pas(["A", "B", "C"], graph)
    assert tie["tie_spots"] == 1 and tie["pas"] == 1.0
    isolated = validation.pas(["A"], sparse.csr_matrix((1, 1)))
    assert isolated["zero_degree_spots"] == 1 and np.isnan(isolated["pas"])


def test_fragmentation_and_permutation_are_deterministic() -> None:
    graph = sparse.block_diag((chain_graph(2), chain_graph(2)), format="csr")
    frame = validation.fragmentation(["A", "A", "A", "B"], graph)
    assert frame.set_index("domain").loc["A", "components"] == 2
    slides = np.array(["S", "S", "S", "S"])
    first = validation.spatial_metrics(["A", "A", "B", "B"], graph, slides, permutations=20, seed=42)[1]
    second = validation.spatial_metrics(["A", "A", "B", "B"], graph, slides, permutations=20, seed=42)[1]
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(validation.ContractError, match="at least 2"):
        validation.spatial_metrics(["A", "A", "B", "B"], graph, slides, permutations=1)


def test_provenance_requires_calibrated_hash_and_reference() -> None:
    good = {"analysis_scope": "exploratory", "reference": "all", "dataset_id": "skin_visium_ssc_paired_cpu_calibrated", "coordinate_strategy": "visium_explicit_scale", "domain_key": validation.DOMAIN_KEY, "neighborhood_valid_key": validation.VALID_KEY, "primary_resolution": 1.0, "accelerator": "cpu", "device": "cpu", "workers": 0, "seed": 42, "input_sha256": validation.EXPECTED_NOVAE_INPUT_SHA256, "checkpoint_sha256": validation.EXPECTED_NOVAE_CHECKPOINT_SHA256, "deterministic_policy": {"requested": True, "effective": True}}
    validation.validate_provenance(good)
    with pytest.raises(validation.ContractError, match="reference"):
        validation.validate_provenance({**good, "reference": "train"})
    with pytest.raises(validation.ContractError, match="hash"):
        validation.validate_provenance({**good, "checkpoint_sha256": "bad"})


def test_shared_valid_filter_has_no_imputation() -> None:
    mask, details = validation.shared_valid_filter([True, False, True], ["a", "b", "c"], ["a", "b", "c"])
    assert mask.tolist() == [True, False, True]
    assert details["imputation"] is False
    with pytest.raises(validation.ContractError, match="sets differ"):
        validation.shared_valid_filter([True], ["a"], ["b"])


def test_run_validation_contract_failures_and_atomic_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAnnData()
    h5ad = tmp_path / "input.h5ad"; post = tmp_path / "post.csv"
    h5ad.write_bytes(b"synthetic h5ad"); fake.post_frame().to_csv(post, index=False)
    monkeypatch.setattr(validation, "_load_anndata", lambda _: fake)
    kwargs = {"expected_h5ad_sha256": validation.sha256(h5ad), "expected_post_sha256": validation.sha256(post), "expected_valid": 18, "expected_invalid": 1, "permutations": 2}
    output = validation.run_validation(h5ad, post, tmp_path / "result", **kwargs)
    assert (output / "top_marker_signature_genes.csv").is_file()
    assert (output / "selected_hvgs.csv").is_file()
    assert (output / "unweighted_per_slide_summary.parquet").is_file()

    missing = FakeAnnData(omit_post_patient=True); missing.post_frame().to_csv(post, index=False)
    with pytest.raises(validation.ContractError, match="post_nmf_obs patient"):
        validation.run_validation(h5ad, post, tmp_path / "missing_metadata", **{**kwargs, "expected_post_sha256": validation.sha256(post)})

    no_l8 = FakeAnnData(omit_l8=True); no_l8.post_frame().to_csv(post, index=False)
    monkeypatch.setattr(validation, "_load_anndata", lambda _: no_l8)
    with pytest.raises(validation.ContractError, match="exactly L0-L8"):
        validation.run_validation(h5ad, post, tmp_path / "missing_domain", **{**kwargs, "expected_post_sha256": validation.sha256(post)})

    fake.post_frame().to_csv(post, index=False)
    monkeypatch.setattr(validation, "_load_anndata", lambda _: fake)
    original_writer = validation._atomic_table
    def fail_writer(frame: pd.DataFrame, path: Path) -> None:
        if path.name == "spatial_metrics_per_slide.parquet":
            raise OSError("synthetic parquet failure")
        original_writer(frame, path)
    monkeypatch.setattr(validation, "_atomic_table", fail_writer)
    with pytest.raises(OSError, match="synthetic parquet"):
        validation.run_validation(h5ad, post, tmp_path / "atomic_failure", **{**kwargs, "expected_post_sha256": validation.sha256(post)})
    assert not (tmp_path / "atomic_failure").exists()
    assert not list(tmp_path.glob(".atomic_failure.*"))


def test_signature_reproducibility_and_confounds() -> None:
    counts = np.array([[10, 0, 2], [9, 1, 1], [0, 10, 2], [1, 9, 1], [8, 2, 0], [7, 3, 1]], dtype=float)
    result = validation.patient_logo_signatures(counts, ["A", "A", "B", "B", "A", "B"], ["P1", "P1", "P2", "P2", "P3", "P3"], genes=["0", "1", "2"], min_heldout=1, min_train=1)
    assert not result.empty
    assert validation.confounding_metrics(["A", "A", "B"], ["P1", "P1", "P2"])["patient_domain_nmi"] >= 0


@pytest.mark.skipif(__import__("os").name == "nt", reason="requires POSIX shell")
def test_launcher_render_conflict_guard(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    env = {**__import__("os").environ, "NOVAE_SPATIAL_REPO_DIR": str(root), "NOVAE_SPATIAL_RUN_ROOT": str(tmp_path / "run"), "NOVAE_SPATIAL_JOB_SCRIPT": str(tmp_path / "run" / "job.sbatch"), "NOVAE_SPATIAL_OUTPUT_DIR": str(tmp_path / "run" / "out"), "NOVAE_SPATIAL_LOG_DIR": str(tmp_path / "run" / "logs")}
    subprocess.run(["bash", str(root / "scripts/submit_novae_spatial_biological_validation.sh"), "--render-only"], env=env, check=True)
    text = (tmp_path / "run" / "job.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=2" in text and "#SBATCH --mem=96gb" in text and 'CUDA_VISIBLE_DEVICES=""' in text
    bad = {**env, "NOVAE_SPATIAL_JOB_NAME": "bad name"}
    assert subprocess.run(["bash", str(root / "scripts/submit_novae_spatial_biological_validation.sh"), "--render-only"], env=bad, capture_output=True).returncode == 2
