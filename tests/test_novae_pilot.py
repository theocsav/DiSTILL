from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("anndata")

import anndata as ad
from scripts.run_novae_pilot import (
    NovaPilotError,
    DEFAULT_DOMAIN_ASSIGNMENT_COVERAGE,
    audit_domain_assignments,
    _build_parser,
    assert_nmf_unchanged,
    expression_audit,
    domain_adjacency,
    domain_proportions,
    graph_diagnostics,
    harmonize_visium_coordinates,
    latent_summary,
    post_inference_expression_audit,
    _science_metrics,
    normalize_resolutions,
    prune_spatial_graph_radius,
    _resolve_model,
    neighbor_distance_calibration,
    validate_input,
    validate_manifest,
    validate_spatial_graph,
    _canonicalize_core_provenance,
    _canonicalize_provenance,
    _h5ad_safe_provenance,
)


def _adata(coords=None, slides=None) -> ad.AnnData:
    coords = np.asarray(coords if coords is not None else [[0, 0], [1, 0], [0, 1], [10, 0]], dtype=float)
    slides = slides or ["A", "A", "A", "B"]
    result = ad.AnnData(np.arange(len(coords) * 2, dtype=float).reshape(len(coords), 2))
    result.obs["slide"] = slides
    result.obs["patient"] = ["P1" if x == "A" else "P2" for x in slides]
    result.obsm["spatial"] = coords
    nmf_labels = [str(index % 2) for index in range(len(coords))]
    result.obs["NMF_factor"] = pd.Categorical(nmf_labels)
    result.obs["dominant_nmf_factor"] = [index % 2 for index in range(len(coords))]
    result.obs["neighborhood_valid"] = True
    return result


def test_manifest_harmonization_uses_different_pixel_diameters_and_preserves_rows():
    data = _adata(coords=[[10, 20], [20, 20], [5, 5], [15, 5]], slides=["A", "A", "B", "B"])
    original = data.obsm["spatial"].copy()
    manifest = pd.DataFrame({"sample_id": ["A", "B"], "spot_diameter_fullres": [50, 100]})
    audit = harmonize_visium_coordinates(data, "slide", manifest, 55.0)
    np.testing.assert_allclose(data.obsm["spatial_original_px"], original)
    np.testing.assert_allclose(data.obsm["spatial"][:2], original[:2] * 1.1)
    np.testing.assert_allclose(data.obsm["spatial"][2:], original[2:] * 0.55)
    assert list(data.obs_names) == ["0", "1", "2", "3"]
    assert audit.set_index("slide").loc["B", "microns_per_pixel"] == pytest.approx(.55)


@pytest.mark.parametrize(
    "manifest",
    [
        pd.DataFrame({"sample_id": ["A"], "spot_diameter_fullres": [50]}),
        pd.DataFrame({"sample_id": ["A", "A", "B"], "spot_diameter_fullres": [50, 50, 100]}),
        pd.DataFrame({"sample_id": ["A", "B"], "spot_diameter_fullres": [0, 100]}),
        pd.DataFrame({"sample_id": ["A", "B", "C"], "spot_diameter_fullres": [50, 100, 50]}),
    ],
)
def test_manifest_errors_are_fail_closed(manifest):
    with pytest.raises(NovaPilotError):
        validate_manifest(manifest, ["A", "B"])


def test_resolution_validation_requires_positive_unique_primary():
    assert normalize_resolutions([0.5, 1.0, 2.0], 1.0) == ([0.5, 1.0, 2.0], 1.0)
    assert normalize_resolutions([0.5, 1.0], 1.0000000000005)[1] == 1.0
    with pytest.raises(NovaPilotError, match="unique"):
        normalize_resolutions([0.5, 0.5], 0.5)
    with pytest.raises(NovaPilotError, match="positive"):
        normalize_resolutions([0.0, 1.0], 1.0)
    with pytest.raises(NovaPilotError, match="included"):
        normalize_resolutions([0.5], 1.0)


def test_neighbor_distance_calibration_uses_physical_medians():
    data = _adata(coords=[[0, 0], [100, 0], [0, 100], [1000, 0]], slides=["A", "A", "A", "B"])
    graph = sparse.csr_matrix([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
    data.obsp["spatial_connectivities"] = graph
    data.obsp["spatial_distances"] = sparse.csr_matrix(
        (np.full(6, 100.0), ([0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1])), shape=graph.shape
    )
    # Slide B has no positive edge and must fail calibration.
    with pytest.raises(NovaPilotError, match="calibration"):
        neighbor_distance_calibration(data, "slide", 100.0)
    data.obs.loc["3", "slide"] = "A"
    data.obs["patient"] = ["P1"] * 4
    graph = sparse.csr_matrix([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=float)
    data.obsp["spatial_connectivities"] = graph
    data.obsp["spatial_distances"] = sparse.csr_matrix(np.where(graph.toarray() > 0, 100.0, 0.0))
    assert neighbor_distance_calibration(data, "slide", 100.0).iloc[0]["pass"]
    data.obsp["spatial_distances"] = data.obsp["spatial_distances"] * 10
    with pytest.raises(NovaPilotError, match="calibration"):
        neighbor_distance_calibration(data, "slide", 100.0)


def test_expression_modes_are_explicit_and_audited():
    data = _adata()
    data.X = sparse.csr_matrix(np.asarray(data.X, dtype=np.int64))
    counts = expression_audit(data, "raw_counts")
    assert counts["mode"] == "raw_counts"
    assert counts["zero_count_rows"]["count"] == 0
    data.X[0, :] = 0
    zero_counts = expression_audit(data, "raw_counts")
    assert zero_counts["zero_count_rows"]["count"] == 1
    assert zero_counts["zero_count_rows"]["obs_ids"] == ["0"]
    assert counts["X"]["sparse"] is True and counts["X"]["integer_like"] is True
    data.X = data.X.astype(float).tocsr()
    data.X.data[0] = -0.5
    with pytest.raises(NovaPilotError, match="raw_counts"):
        expression_audit(data, "raw_counts")
    data.X.data[0] = -0.5
    preprocessed = expression_audit(data, "preprocessed")
    assert preprocessed["mode"] == "preprocessed" and preprocessed["X"]["finite"] is True
    data.X.data[0] = np.nan
    with pytest.raises(NovaPilotError, match="non-finite"):
        expression_audit(data, "preprocessed")


def test_duplicate_and_nonfinite_coordinates_rejected():
    with pytest.raises(NovaPilotError, match="duplicate"):
        validate_input(_adata(coords=[[0, 0], [0, 0], [1, 1], [2, 2]]), "slide")
    with pytest.raises(NovaPilotError, match="finite"):
        validate_input(_adata(coords=[[0, 0], [np.nan, 1], [1, 1], [2, 2]]), "slide")


def test_group_boundary_rejects_ambiguous_slide_mapping_and_records_valid_mapping():
    ambiguous = _adata(slides=["A", "A", "B", "B"])
    ambiguous.obs["patient"] = ["P1", "P2", "P3", "P3"]
    with pytest.raises(NovaPilotError, match="exactly one"):
        validate_input(ambiguous, "slide", "patient")
    valid = _adata(slides=["A", "B", "C", "C"])
    valid.obs["patient"] = ["P1", "P1", "P1", "P1"]
    audit = validate_input(valid, "slide", "patient")
    assert audit["slide_group_mapping"] == {"A": "P1", "B": "P1", "C": "P1"}


def test_graph_cross_slide_rejected_and_diagnostics():
    data = _adata()
    data.obsp["spatial_connectivities"] = sparse.csr_matrix(
        [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=float
    )
    data.obsp["spatial_distances"] = data.obsp["spatial_connectivities"].copy()
    assert validate_spatial_graph(data, "slide")["cross_slide_edges"] == 0
    diagnostics = graph_diagnostics(data, "slide")
    assert diagnostics.set_index("slide").loc["A", "undirected_edges"] == 3
    assert diagnostics.set_index("slide").loc["A", "connected_components"] == 1
    assert diagnostics.set_index("slide").loc["B", "zero_degree_count"] == 1
    explicit_zero = data.obsp["spatial_connectivities"].copy()
    explicit_zero[3, 3] = 0.0
    assert explicit_zero.nnz > data.obsp["spatial_connectivities"].nnz
    data.obsp["spatial_connectivities"] = explicit_zero
    explicit_diagnostics = graph_diagnostics(data, "slide")
    assert explicit_diagnostics.set_index("slide").loc["B", "zero_degree_count"] == 1
    assert data.obsp["spatial_connectivities"].nnz == explicit_zero.nnz
    graph = data.obsp["spatial_connectivities"].tolil()
    graph[0, 3] = 1
    data.obsp["spatial_connectivities"] = graph.tocsr()
    with pytest.raises(NovaPilotError, match="cross-slide"):
        validate_spatial_graph(data, "slide")


def test_radius_pruning_uses_materialized_microns_and_shared_scalar_scale():
    data = _adata(coords=[[0, 0], [50, 0], [150, 0], [0, 0]], slides=["A", "A", "A", "B"])
    graph = sparse.csr_matrix(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=float
    )
    explicit_graph = graph.copy()
    explicit_graph[2, 2] = 0.0
    data.obsp["spatial_connectivities"] = explicit_graph
    data.obsp["spatial_distances"] = graph.copy()
    result = prune_spatial_graph_radius(data, "slide", 75, 1.0)
    assert result["removed_undirected_edges"] == 1
    assert result["pre_zero_degree_count"] == 1 and result["post_zero_degree_count"] == 2
    assert result["per_slide"][0]["pre_zero_degree_count"] == 0
    assert result["per_slide"][0]["post_zero_degree_count"] == 1
    assert data.obsp["spatial_connectivities"].nnz == 2
    assert data.obsp["spatial_distances"].nnz == 2
    data.obsp["spatial_connectivities"] = graph.copy()
    data.obsp["spatial_distances"] = graph.copy()
    result = prune_spatial_graph_radius(data, "slide", 150, 2.0)
    assert result["removed_undirected_edges"] == 1
    assert result["effective_scale_to_microns"] == 2.0
    with pytest.raises(NovaPilotError, match="destroyed all edges"):
        prune_spatial_graph_radius(data, "slide", 1, 1.0)


def test_local_model_revision_is_not_claimed_verified(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    _, provenance = _resolve_model(str(model), "requested-revision")
    assert provenance["requested_revision"] == "requested-revision"
    assert provenance["resolved_revision"] is None
    assert provenance["revision_verified"] is False
    assert provenance["files_sha256"]


def test_remote_model_revision_remains_verified(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = type("HfApi", (), {"model_info": lambda self, source, revision: types.SimpleNamespace(sha="resolved-sha")})
    hub.snapshot_download = lambda source, revision: str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    _, provenance = _resolve_model("remote/model", "requested-revision")
    assert provenance["resolved_revision"] == "resolved-sha"
    assert provenance["revision_verified"] is True
    assert provenance["files_sha256"]


def test_domain_adjacency_counts_undirected_edges_and_normalizes():
    data = _adata()
    data.obs["domain"] = ["d1", "d2", "d1", "d1"]
    data.obsp["spatial_connectivities"] = sparse.csr_matrix(
        [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=float
    )
    adjacency = domain_adjacency(data, "slide", "domain")
    pairs = adjacency.set_index(["domain_a", "domain_b"])
    assert pairs.loc[("d1", "d1"), "edge_count"] == 1
    assert pairs.loc[("d1", "d2"), "edge_count"] == 2
    assert pairs["edge_proportion"].sum() == pytest.approx(1.0)


def test_invalid_neighborhoods_are_reported_and_excluded_from_summaries():
    data = _adata(coords=[[0, 0], [1, 0], [2, 0], [3, 0]], slides=["A"] * 4)
    data.obs["neighborhood_valid"] = [True, True, True, False]
    data.obs["domain"] = pd.Categorical(["d1", "d2", "d1", np.nan])
    data.obs["novae_leaves"] = pd.Categorical(["D1", "D2", "D1", np.nan])
    data.obsm["novae_latent"] = np.asarray([[1, 2], [3, 4], [5, 6], [0, 0]], dtype=float)
    audit = audit_domain_assignments(data, "slide", "domain")
    assert audit["minimum_coverage"] == DEFAULT_DOMAIN_ASSIGNMENT_COVERAGE
    assert data.obs["novae_leaves"].isna().sum() == 1
    assert audit["overall"] == {
        **audit["overall"], "total": 4, "valid_neighborhood": 3,
        "assigned": 3, "unassigned": 1, "coverage": pytest.approx(0.75),
    }
    proportions = domain_proportions(data, "slide", "domain")
    assert set(proportions["domain"]) == {"d1", "d2"}
    assert proportions.iloc[0]["total"] == 4 and proportions.iloc[0]["assigned"] == 3
    data.obsp["spatial_connectivities"] = sparse.csr_matrix(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float
    )
    adjacency = domain_adjacency(data, "slide", "domain")
    assert adjacency.iloc[0]["total_edges"] == 3
    assert adjacency.iloc[0]["used_edges"] == 2 and adjacency.iloc[0]["excluded_edges"] == 1
    latent = latent_summary(data, "slide", "novae_latent")
    assert latent.iloc[0]["total"] == 4 and latent.iloc[0]["valid"] == 3 and latent.iloc[0]["excluded"] == 1
    assert latent.iloc[0]["latent_0_mean"] == pytest.approx(3)


def test_domain_assignment_coverage_is_enforced_per_slide_even_when_overall_passes():
    data = _adata(coords=[[index, 0] for index in range(10)], slides=["A"] * 8 + ["B"] * 2)
    data.obs["neighborhood_valid"] = [True] * 8 + [False, False]
    data.obs["domain"] = pd.Categorical(["d1"] * 8 + [np.nan, np.nan])
    with pytest.raises(NovaPilotError, match="failing slides.*B.*total=2.*assigned=0.*coverage=0.0000"):
        audit_domain_assignments(data, "slide", "domain", 0.70)


def test_science_metrics_reject_nonfinite_official_results(monkeypatch):
    data = _adata()
    data.obs["domain"] = pd.Categorical(["d1", "d2", "d1", "d2"])
    module = types.ModuleType("novae")
    module.monitor = types.SimpleNamespace(
        mean_fide_score=lambda *args, **kwargs: np.nan,
        jensen_shannon_divergence=lambda *args, **kwargs: np.inf,
    )
    monkeypatch.setitem(sys.modules, "novae", module)
    with pytest.raises(NovaPilotError, match="non-finite FIDE"):
        _science_metrics(module, data, "slide", "domain")


def test_domain_assignment_validation_rejects_valid_missing_and_low_coverage():
    data = _adata()
    data.obs["neighborhood_valid"] = [True, True, False, False]
    data.obs["domain"] = pd.Categorical(["d1", np.nan, np.nan, np.nan])
    with pytest.raises(NovaPilotError, match="valid neighborhoods"):
        audit_domain_assignments(data, "slide", "domain", 0.0)
    data.obs["neighborhood_valid"] = [True, False, False, False]
    data.obs["domain"] = pd.Categorical(["d1", np.nan, np.nan, np.nan])
    with pytest.raises(NovaPilotError, match="below minimum"):
        audit_domain_assignments(data, "slide", "domain", 0.70)
    data.obs["domain"] = ["nan", np.nan, np.nan, np.nan]
    with pytest.raises(NovaPilotError, match="valid neighborhoods"):
        audit_domain_assignments(data, "slide", "domain", 0.0)


def test_domain_assignment_coverage_parser_and_launcher_override(tmp_path):
    args = _build_parser().parse_args([
        "--input-h5ad", "in.h5ad", "--output-dir", "out", "--dataset-id", "x",
        "--slide-key", "slide", "--expression-mode", "raw_counts",
        "--coordinate-strategy", "materialized_microns",
    ])
    assert args.min_domain_assignment_coverage == pytest.approx(0.70)
    args = _build_parser().parse_args([
        "--input-h5ad", "in.h5ad", "--output-dir", "out", "--dataset-id", "x",
        "--slide-key", "slide", "--expression-mode", "raw_counts",
        "--coordinate-strategy", "materialized_microns", "--min-domain-assignment-coverage", "0.85",
    ])
    assert args.min_domain_assignment_coverage == pytest.approx(0.85)
    script = Path(__file__).parents[1] / "scripts" / "submit_novae_skin_pilot.sh"
    env = os.environ.copy()
    env.update({"NOVAE_REPO_DIR": str(tmp_path / "repo"), "NOVAE_RUN_ROOT": str(tmp_path / "run"),
                "NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE": "0.85"})
    subprocess.run(["bash", str(script), "--render-only"], env=env, check=True, capture_output=True, text=True)
    generated = (tmp_path / "run" / "submit_novae_skin_pilot.sbatch").read_text()
    assert "--min-domain-assignment-coverage 0.85" in generated
    bad = dict(env, NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE="1.1")
    assert subprocess.run(["bash", str(script), "--render-only"], env=bad, capture_output=True).returncode == 2


def test_domain_and_latent_summaries_and_nmf_preservation():
    data = _adata()
    data.obs["domain"] = ["d1", "d2", "d1", "d2"]
    proportions = domain_proportions(data, "slide", "domain")
    assert proportions.groupby("slide")["proportion"].sum().tolist() == [pytest.approx(1), pytest.approx(1)]
    data.obsm["novae_latent"] = np.arange(8, dtype=float).reshape(4, 2)
    summary = latent_summary(data, "slide", "novae_latent")
    assert summary.set_index("slide").loc["A", "latent_0_mean"] == pytest.approx(2)
    data.obs["NMF_factor"] = data.obs["NMF_factor"].astype(object)
    data.obs.loc["0", "NMF_factor"] = np.nan
    snapshot = {key: {"values": data.obs[key].astype(object).tolist(), "dtype": str(data.obs[key].dtype), "categories": data.obs[key].cat.categories.astype(str).tolist() if hasattr(data.obs[key].dtype, "categories") else None} for key in ("NMF_factor", "dominant_nmf_factor")}
    assert_nmf_unchanged(data, snapshot)
    data.obs["NMF_factor"] = data.obs["NMF_factor"].astype(str)
    data.obs.loc["0", "NMF_factor"] = "changed"
    with pytest.raises(NovaPilotError):
        assert_nmf_unchanged(data, snapshot)


def _cli_args(input_path, output_path, model_path, graph_radius_um=None):
    values = [
        "--input-h5ad", str(input_path), "--output-dir", str(output_path),
        "--dataset-id", "synthetic", "--slide-key", "slide", "--group-key", "patient",
        "--technology", "visium", "--expression-mode", "raw_counts",
        "--model-source", str(model_path), "--resolutions", "0.5", "1.0", "2.0",
        "--primary-resolution", "1.0", "--expected-neighbor-distance-um", "100",
        "--accelerator", "cpu",
        "--workers", "0", "--coordinate-strategy", "materialized_microns",
    ]
    if graph_radius_um is not None:
        values.extend(["--graph-radius-um", str(graph_radius_um)])
    return _build_parser().parse_args(values)


def _fake_novae(monkeypatch, late_failure=False):
    module = types.ModuleType("novae")
    calls = {"compute": 0, "assign": [], "fide": [], "jsd": []}
    module.__version__ = "1.1.1"
    module.settings = types.SimpleNamespace(scale_to_microns=None, auto_preprocessing=None)

    def spatial_neighbors(adata, *, slide_key, technology, **kwargs):
        # This deliberately rejects the old reset_slide_ids=False workaround.
        assert "reset_slide_ids" not in kwargs
        labels = adata.obs[slide_key].astype(str).to_numpy()
        rows, cols = [], []
        for i in range(len(labels) - 1):
            if labels[i] == labels[i + 1]:
                rows.extend([i, i + 1]); cols.extend([i + 1, i])
        matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(adata.n_obs, adata.n_obs))
        adata.obsp["spatial_connectivities"] = matrix
        adata.obsp["spatial_distances"] = sparse.csr_matrix(
            (np.full(len(rows), 100.0), (rows, cols)), shape=(adata.n_obs, adata.n_obs)
        )
        # Mimic NOVAE's default internal slide-ID initialization while leaving
        # the user-provided boundary column untouched.
        adata.obs["novae_sid"] = pd.Categorical([f"internal_{value}" for value in labels])

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        hparams = {"n_hops_local": 2, "n_hops_view": 2}

        def compute_representations(self, adata, **kwargs):
            calls["compute"] += 1
            assert kwargs["zero_shot"] is True and kwargs["reference"] == "all"
            adata.layers["counts"] = adata.X.copy()
            adata.X = np.asarray(adata.X, dtype=float) / 2.0
            adata.obsm["novae_latent"] = np.ones((adata.n_obs, 2))
            adata.obs["neighborhood_valid"] = pd.Series(True, index=adata.obs_names)

        def assign_domains(self, adata, *, resolution):
            calls["assign"].append(resolution)
            if late_failure:
                return "missing_domain"
            key = f"novae_domains_res{resolution}"
            adata.obs[key] = pd.Categorical([f"D{int(float(resolution) * 10)}"] * adata.n_obs)
            return key

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text("{}")

    def mean_fide_score(adata, *, obs_key, slide_key):
        calls["fide"].append((obs_key, slide_key))
        return 0.8

    def jensen_shannon_divergence(adata, *, obs_key, slide_key):
        calls["jsd"].append((obs_key, slide_key))
        return 0.2

    module.monitor = types.SimpleNamespace(
        mean_fide_score=mean_fide_score,
        jensen_shannon_divergence=jensen_shannon_divergence,
    )
    module.spatial_neighbors = spatial_neighbors
    module.Novae = FakeModel
    monkeypatch.setitem(sys.modules, "novae", module)
    return calls


def test_mocked_end_to_end_initializes_fresh_slides_and_embedded_core_matches_external(tmp_path, monkeypatch):
    data = _adata(coords=[[0, 0], [1, 0], [10, 0], [11, 0]], slides=["A", "A", "B", "B"])
    data.X = np.ones((4, 2), dtype=np.int64)
    input_path = tmp_path / "input.h5ad"
    data.write_h5ad(input_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"fake")
    calls = _fake_novae(monkeypatch)
    from scripts import run_novae_pilot as pilot
    output_path = tmp_path / "out"
    assert pilot._run(_cli_args(input_path, output_path, model_path, graph_radius_um=100)) == 0
    assert output_path.exists()
    assert not list(output_path.glob("*.partial"))
    assert not list(tmp_path.glob(".out.staging-*"))
    envelope = json.loads((output_path / "novae_provenance_synthetic.json").read_text())
    coordinate_audit = json.loads((output_path / "novae_coordinate_audit_synthetic.json").read_text())
    written = ad.read_h5ad(output_path / "novae_synthetic_zero_shot.h5ad")
    assert written.uns["novae_pilot_provenance"] == envelope["run"]
    per_slide = envelope["run"]["radius_pruning"]["per_slide"]
    assert set(per_slide) == {"A", "B"}
    assert all("slide" not in record for record in per_slide.values())
    assert per_slide["A"]["obs"] == 2
    assert isinstance(coordinate_audit["graph_radius_pruning"]["per_slide"], list)
    assert coordinate_audit["graph_radius_pruning"]["per_slide"][0]["slide"] == "A"
    assert written.obs["slide"].astype(str).tolist() == ["A", "A", "B", "B"]
    assert envelope["run"]["novae_auto_preprocessing"] is True
    assert envelope["run"]["auto_preprocessing_applied"] is True
    assert envelope["run"]["counts_layer_created"] is True
    assert envelope["run"]["counts_layer_preserved"] is True
    assert calls["compute"] == 1
    assert calls["assign"] == [0.5, 1.0, 2.0]
    assert len(calls["fide"]) == len(calls["jsd"]) == 3
    assert all(slide_key == "slide" for _, slide_key in calls["fide"] + calls["jsd"])
    assert all(obs_key.startswith("novae_domains_res") for obs_key, _ in calls["fide"] + calls["jsd"])
    assert envelope["run"]["representation_computation_count"] == 1
    assert envelope["run"]["primary_resolution"] == 1.0
    assert envelope["run"]["zero_shot_model_artifact"]["note"].find("cohort-derived") >= 0
    assert envelope["run"]["checkpoint_hparams"] == {"n_hops_local": 2, "n_hops_view": 2}
    assert len(envelope["run"]["domain_keys"]) == 3
    assert (output_path / "novae_domain_resolution_summary_synthetic.csv").exists()
    assert (output_path / "novae_science_metrics_synthetic.csv").exists()
    for token in ("0p5", "1", "2"):
        assert (output_path / f"novae_domain_adjacency_synthetic_res-{token}.csv").exists()
        assert (output_path / f"novae_domain_proportions_synthetic_res-{token}.csv").exists()
    assert (output_path / "novae_zero_shot_model").exists()


@pytest.mark.parametrize("normalizer", [_canonicalize_provenance, _h5ad_safe_provenance])
def test_provenance_normalizers_reject_mapping_key_collisions(normalizer):
    with pytest.raises(NovaPilotError, match="mapping-key collision.*1"):
        normalizer({1: "a", "1": "b"})
    with pytest.raises(NovaPilotError, match=r"provenance\.nested.*mapping-key collision"):
        normalizer({"nested": {1: "a", "1": "b"}})


@pytest.mark.parametrize("normalizer", [_canonicalize_provenance, _h5ad_safe_provenance])
def test_provenance_normalizers_reject_nullable_sequence_members(normalizer):
    with pytest.raises(NovaPilotError, match=r"provenance\[1\].*None"):
        normalizer(["present", None])
    with pytest.raises(NovaPilotError, match=r"provenance\.nested\[0\].*None"):
        normalizer({"nested": (None,)})


def test_core_provenance_normalizes_nested_record_and_mixed_lists():
    provenance = {
        "radius_pruning": {"per_slide": [
            {"slide": "B", "obs": 2}, {"slide": "A", "obs": 3},
        ]},
        "mixed": [1, "two", {"nested": True}],
    }
    result = _canonicalize_core_provenance(provenance)
    assert result["radius_pruning"]["per_slide"] == {
        "A": {"obs": 3}, "B": {"obs": 2},
    }
    assert result["mixed"] == {
        "0": 1, "1": "two", "2": {"nested": True},
    }


def test_late_failure_removes_final_output_directory(tmp_path, monkeypatch):
    data = _adata(coords=[[0, 0], [1, 0], [10, 0], [11, 0]], slides=["A", "A", "B", "B"])
    data.X = np.ones((4, 2), dtype=np.int64)
    input_path = tmp_path / "input.h5ad"
    data.write_h5ad(input_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _fake_novae(monkeypatch, late_failure=True)
    from scripts import run_novae_pilot as pilot
    output_path = tmp_path / "late-out"
    with pytest.raises(NovaPilotError):
        pilot._run(_cli_args(input_path, output_path, model_path))
    assert not output_path.exists()
    assert not list(tmp_path.glob(".late-out.staging-*"))


def test_hpg_launcher_render_only_is_safe_and_omits_empty_partition(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "submit_novae_skin_pilot.sh"
    env = os.environ.copy()
    env.update({"NOVAE_REPO_DIR": str(tmp_path / "repo"), "NOVAE_RUN_ROOT": str(tmp_path / "run")})
    result = subprocess.run(["bash", str(script), "--render-only"], env=env, text=True, capture_output=True, check=True)
    generated = (tmp_path / "run" / "submit_novae_skin_pilot.sbatch").read_text()
    assert "#SBATCH --qos=kejun.huang" in generated
    assert "#SBATCH --partition=" not in generated
    assert "--graph-radius-um 100" in generated
    assert not (tmp_path / "run" / "skin_visium_ssc").exists()
    assert "Rendered sbatch script" in result.stdout


def test_transformed_raw_counts_without_preserved_counts_is_rejected():
    data = _adata()
    original = data.X.copy()
    data.X = np.asarray(data.X, dtype=float) / 2.0
    with pytest.raises(NovaPilotError, match="without preserving"):
        post_inference_expression_audit(data, "raw_counts", original, True, False)


def test_legacy_resolution_is_single_and_primary_defaults_to_alias():
    args = _build_parser().parse_args(["--input-h5ad", "in.h5ad", "--output-dir", "out",
                                       "--dataset-id", "x", "--slide-key", "slide",
                                       "--expression-mode", "raw_counts", "--coordinate-strategy",
                                       "materialized_microns", "--resolution", "0.5"])
    assert args.resolutions is None and args.primary_resolution is None
    assert normalize_resolutions([args.resolution], args.primary_resolution or args.resolution) == ([0.5], 0.5)


def test_exploratory_semantics_are_explicit_in_help():
    text = _build_parser().description + "\n" + _build_parser().format_help()
    assert "exploratory" in text.lower()
    assert "confirmatory" in text.lower()
    assert "--expression-mode" in text
