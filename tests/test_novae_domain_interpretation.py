from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_novae_domain_interpretation as interpretation


def test_pseudobulk_cpm_denominator_uses_all_genes() -> None:
    counts = np.asarray([[10, 0], [10, 0], [5, 100], [5, 100]], float)
    result = interpretation.patient_domain_signatures(counts, ["A"] * 2 + ["B"] * 2, ["P"] * 4, [0], min_domain=2, min_background=2, gene_names=["g0", "unselected"])
    observed = float(result[(result.domain == "A") & (result.gene == "g0")].log2_cpm_domain_minus_background.iloc[0])
    expected = np.log2(1_000_000 + 1) - np.log2((10 / 210 * 1_000_000) + 1)
    assert np.isclose(observed, expected)


def test_patient_aware_pseudobulk_and_marker_gate() -> None:
    counts = np.asarray([[10, 0, 1], [10, 0, 1], [10, 0, 1], [10, 0, 1], [10, 0, 1], [0, 10, 1], [0, 10, 1], [0, 10, 1], [0, 10, 1], [0, 10, 1]], float)
    signatures = interpretation.patient_domain_signatures(
        counts, ["A"] * 5 + ["B"] * 5, ["P"] * 10, [0, 1, 2], min_domain=5, min_background=5,
        gene_names=["g0", "g1", "g2"], arm="nmf",
    )
    assert len(signatures) == 6 and signatures.domain.nunique() == 2
    aggregate = interpretation.aggregate_patient_signatures(signatures, min_patients=1)
    assert {"median_within_patient_rank_fraction", "rank_IQR", "rank_stability"}.issubset(aggregate.columns)
    assert set(interpretation.top_positive_markers(aggregate).gene) >= {"g0", "g1"}
    assert interpretation.eligible_detected_genes(counts).tolist() == [0, 1, 2]


def test_gmt_ora_bh_and_fixed_programs() -> None:
    aggregate = pd.DataFrame({"arm": ["nmf", "nmf"], "domain": ["0", "0"], "gene": ["A", "B"], "patient_count": [3, 3], "median_logFC": [2.0, 1.0], "q25_logFC": [1, 0], "q75_logFC": [3, 2], "IQR_logFC": [2, 2], "fraction_positive": [1, 1], "rank_stability": [1, 1]})
    programs = pd.DataFrame({"category": ["x"], "program": ["p"], "source_id": ["test"], "genes": [("A", "C")]})
    scores = interpretation.score_programs(aggregate, programs)
    assert scores.iloc[0].covered_genes == 1 and scores.iloc[0].coverage == .5
    ora = interpretation.hallmark_ora(aggregate, ["A", "B", "C"], {"SET": {"A", "C"}})
    assert ora.iloc[0].overlap_count == 1 and ora.iloc[0].q_value == ora.iloc[0].p_value


def test_matching_orientation_ties_and_agreement() -> None:
    sim = pd.DataFrame({"left_arm": ["nmf"] * 4, "left_domain": ["0", "0", "1", "1"], "right_arm": ["novae"] * 4, "right_domain": ["L0", "L1", "L0", "L1"], "common_gene_count": [3] * 4, "spearman": [1, 1, 1, 1]})
    first = interpretation.hungarian_matching(sim)
    second = interpretation.hungarian_matching(sim)
    pd.testing.assert_frame_equal(first, second)
    assert set(map(tuple, first[["nmf_domain", "novae_domain"]].to_numpy())) == {("0", "L0"), ("1", "L1")}
    contingency, agreement = interpretation.spot_agreement(["0", "0", "1", "1"], ["L0", "L0", "L1", "L1"], ["P1", "P1", "P2", "P2"])
    assert contingency.spots.sum() == 4 and agreement.iloc[0].ARI == 1


def _prevalence() -> pd.DataFrame:
    rows = []
    for i in range(14):
        disease = "healthy" if i < 4 else "systemic_sclerosis"
        for domain in [str(j) for j in range(9)]:
            rows.append({"arm": "nmf", "patient": f"P{i}", "disease": disease, "domain": domain, "proportion": (i + int(domain)) / 100})
    return pd.DataFrame(rows)


def test_prevalence_retains_absent_domain_zeros() -> None:
    coverage = pd.DataFrame({"patient": [f"P{i}" for i in range(14)], "disease": ["healthy"] * 4 + ["systemic_sclerosis"] * 10, "nmf_factor": ["0"] * 14, "novae_domain": ["L0"] * 14})
    frame = interpretation.complete_patient_prevalence(coverage)
    assert len(frame) == 14 * 9 * 2
    assert (frame.query("arm == 'nmf' and domain == '1'").spots == 0).all()


def test_exact_1001_and_bootstrap_determinism() -> None:
    frame = _prevalence()
    result = interpretation.exact_prevalence_test(frame)
    assert len(result) == 9 and set(result.permutations) == {1001}
    first = interpretation.prevalence_bootstrap(frame[frame.domain == "0"], reps=200, seed=42)
    second = interpretation.prevalence_bootstrap(frame[frame.domain == "0"], reps=200, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_maps_require_safe_complete_slide_set(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows = []
    for i in range(14):
        rows.append({"slide": f"S{i}", "x": 0.0, "y": 0.0, "nmf_factor": "0", "novae_domain": "L0"})
    matching = pd.DataFrame({"nmf_domain": ["0"], "novae_domain": ["L0"], "spearman": [1.0]})
    files = interpretation._maps(pd.DataFrame(rows), tmp_path, matching)
    assert len([x for x in files if x.endswith(".png")]) == 14 and (tmp_path / "domain_maps.pdf").stat().st_size > 0
    unsafe = pd.DataFrame(rows); unsafe.loc[0, "slide"] = "../escape"
    with pytest.raises(interpretation.ContractError, match="unsafe"):
        interpretation._maps(unsafe, tmp_path / "unsafe", matching)


def test_end_to_end_synthetic_run_and_atomic_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAnnData:
        pass
    fake = FakeAnnData(); rows = []
    for patient_index in range(14):
        disease = "healthy" if patient_index < 4 else "systemic_sclerosis"
        for domain_index in range(9):
            for replicate in range(9):
                rows.append((f"cell_{patient_index}_{domain_index}_{replicate}", f"S{patient_index}", f"P{patient_index}", disease, str(domain_index), f"L{domain_index}"))
    ids, slides, patients, diseases, nmf, novae = zip(*rows, strict=True)
    fake.obs_names = pd.Index(ids); fake.var_names = pd.Index(["G0", "G1", "G2", "G3", "G4"]); fake.n_obs = len(rows)
    fake.obs = pd.DataFrame({"unique_cell_id": ids, "sample_id": slides, "patient": patients, "Disease_State": diseases, "neighborhood_valid": [True] * len(rows), "novae_domains_res1.0": novae}, index=ids)
    fake.obsm = {"spatial": np.arange(len(rows) * 2, dtype=float).reshape(len(rows), 2)}; fake.layers = {"counts": np.ones((len(rows), 5), dtype=np.int32)}
    for row, (_, _, _, _, domain, _) in enumerate(rows): fake.layers["counts"][row, int(domain) % 5] += 5
    h5ad, post, validation = tmp_path / "input.h5ad", tmp_path / "post.csv", tmp_path / "validation"; h5ad.write_bytes(b"fake"); pd.DataFrame({"unique_cell_id": ids, "NMF_factor": nmf, "patient": patients, "Disease_State": diseases, "sample_id": slides}).to_csv(post, index=False); validation.mkdir()
    prior_obs = pd.DataFrame({"unique_cell_id": ids, "x": fake.obsm["spatial"][:, 0], "y": fake.obsm["spatial"][:, 1]}); prior_obs.to_parquet(validation / "shared_observation_graph_contract.parquet", index=False)
    pd.DataFrame({"arm": ["nmf"] * 9 + ["novae"] * 9, "domain": [str(i) for i in range(9)] * 2, "same_domain_spearman": [1.0] * 18, "best_other_domain_spearman": [0.0] * 18, "margin": [1.0] * 18}).to_parquet(validation / "patient_logo_signatures.parquet", index=False)
    (validation / "graph_contract.json").write_text(__import__("json").dumps({"coordinates": "identical shared x/y", "expression_source": "layers['counts']", "nodes": len(rows)}))
    prior_files = ["shared_observation_graph_contract.parquet", "patient_logo_signatures.parquet", "graph_contract.json"]
    manifest = {"contract": "novae_spatial_biological_validation", "inputs": {"novae_h5ad": {"path": str(h5ad), "sha256": interpretation.sha256(h5ad)}, "post_nmf_obs": {"path": str(post), "sha256": interpretation.sha256(post)}}, "outputs": {name: interpretation.sha256(validation / name) for name in prior_files}}
    (validation / "manifest.json").write_text(__import__("json").dumps(manifest))
    monkeypatch.setattr(interpretation, "_load_anndata", lambda _: fake)
    out = interpretation.run_interpretation(h5ad, post, validation, tmp_path / "out", expected_h5ad_sha256=interpretation.sha256(h5ad), expected_post_sha256=interpretation.sha256(post), expected_valid=len(rows), expected_invalid=0, bootstrap_reps=10)
    assert len(list(out.glob("map_*.png"))) == 14 and (out / "domain_maps.pdf").stat().st_size > 0
    output_manifest = __import__("json").loads((out / "manifest.json").read_text())
    assert output_manifest["outputs"] and not list(tmp_path.glob(".out.*"))


def test_resources_and_path_traversal_launcher(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    programs = interpretation.read_programs(root / interpretation.PROGRAM_RESOURCE)
    assert len(programs) >= 30 and programs.genes.map(len).min() >= 2
    assert interpretation.sha256(root / interpretation.HALLMARK_RESOURCE) == interpretation.HALLMARK_SHA256
    assert interpretation.sha256(root / interpretation.PROGRAM_RESOURCE) == interpretation.PROGRAM_SHA256
    assert len(interpretation.parse_gmt(root / interpretation.HALLMARK_RESOURCE)) == 50
    assert not list(tmp_path.glob(".out.*"))


@pytest.mark.skipif(__import__("os").name == "nt", reason="requires POSIX shell")
def test_launcher_render_resources(tmp_path: Path) -> None:
    import os
    import subprocess
    root = Path(__file__).parents[1]
    env = {**os.environ, "NOVAE_INTERPRET_REPO_DIR": str(root), "NOVAE_INTERPRET_H5AD": str(tmp_path / "input.h5ad"), "NOVAE_INTERPRET_POST_NMF_OBS": str(tmp_path / "post.csv"), "NOVAE_INTERPRET_VALIDATION_DIR": str(tmp_path / "validation"), "NOVAE_INTERPRET_RUN_ROOT": str(tmp_path / "run"), "NOVAE_INTERPRET_JOB_SCRIPT": str(tmp_path / "run" / "job.sbatch"), "NOVAE_INTERPRET_OUTPUT_DIR": str(tmp_path / "run" / "out"), "NOVAE_INTERPRET_LOG_DIR": str(tmp_path / "run" / "logs")}
    subprocess.run(["bash", str(root / "scripts/submit_novae_domain_interpretation.sh"), "--render-only"], env=env, check=True)
    text = (tmp_path / "run" / "job.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=2" in text and "#SBATCH --mem=96gb" in text and "CUDA_VISIBLE_DEVICES=\"\"" in text
    assert "run_novae_domain_interpretation.py" in text
    subprocess.run([__import__("sys").executable, str(root / "scripts/run_novae_domain_interpretation.py"), "--help"], check=True, capture_output=True)
