from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")
from scripts.compare_novae_runs import NovaComparisonError, build_parser, compare_runs, main


def _pair(tmp_path: Path, *, altered_graph: bool = False):
    base = ad.AnnData(np.ones((4, 3), dtype=np.int64))
    base.obs_names = ["a", "b", "c", "d"]
    base.var_names = ["g1", "g2", "g3"]
    for obj in (base,):
        obj.obs["sample_id"] = ["A", "A", "B", "B"]
        obj.obs["neighborhood_valid"] = [True, True, True, True]
        obj.obs["novae_domains_res0.5"] = pd.Categorical(["x", "x", "y", "y"])
        obj.obsm["novae_latent"] = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
        obj.obsp["spatial_connectivities"] = sparse.csr_matrix(
            ([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        )
    sensitivity = base.copy()
    sensitivity.obsm["novae_latent"] = sensitivity.obsm["novae_latent"] + 0.1
    if altered_graph:
        sensitivity.obsp["spatial_connectivities"] = sparse.csr_matrix(
            ([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
        )
    baseline_h5ad, sensitivity_h5ad = tmp_path / "baseline.h5ad", tmp_path / "sensitivity.h5ad"
    base.write_h5ad(baseline_h5ad); sensitivity.write_h5ad(sensitivity_h5ad)
    run = {
        "latent_key": "novae_latent", "science_metrics": {"0p5": {"FIDE": 0.8, "JSD": 0.2}},
        "input_sha256": "input", "checkpoint_sha256": "checkpoint", "model_revision": "revision",
        "seed": 42, "expression_mode": "raw_counts", "requested_resolutions": {"0p5": 0.5},
        "primary_resolution": 0.5, "technology": "visium",
        "distance_qc_expected_um": 100.0, "distance_qc_relative_tolerance": 0.5,
        "minimum_domain_assignment_coverage": 0.70, "slide_key": "sample_id", "group_key": "patient",
        "reference": "all", "inference_mode": "zero_shot", "accelerator": "cpu", "workers": 0,
        "coordinate_strategy": "visium_manifest",
        "radius_pruning": {"applied": True, "removed_undirected_edges": 0},
    }
    baseline_manifest, sensitivity_manifest = tmp_path / "baseline.json", tmp_path / "sensitivity.json"
    sensitivity_run = {**run, "coordinate_strategy": "visium_explicit_scale", "radius_pruning": None}
    baseline_manifest.write_text(json.dumps({"run": run})); sensitivity_manifest.write_text(json.dumps({"run": sensitivity_run}))
    return baseline_h5ad, sensitivity_h5ad, baseline_manifest, sensitivity_manifest


def test_comparison_reports_identity_and_does_not_load_expression(tmp_path):
    args = _pair(tmp_path)
    output = compare_runs(*args, tmp_path / "comparison")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert report["acceptance"]["graph_identity"]
    assert report["acceptance"]["overall_accepted"]
    assert report["latent"]["common_valid_finite_rows"] == 4
    assert (output / "domain_comparison.csv").is_file()
    latent_slides = pd.read_csv(output / "latent_per_slide_comparison.csv")
    coverage = pd.read_csv(output / "coverage_comparison.csv")
    assert set(latent_slides["slide"]) == {"A", "B"}
    assert set(coverage["side"]) == {"baseline", "sensitivity"}
    assert not list(tmp_path.glob(".comparison.*.partial"))


def test_science_missing_resolution_is_not_filtered_and_rejects_acceptance(tmp_path):
    args = _pair(tmp_path)
    missing = json.loads(args[3].read_text())
    missing["run"]["science_metrics"] = {}
    args[3].write_text(json.dumps(missing))
    output = compare_runs(*args, tmp_path / "missing-science")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["acceptance"]["science_fide_jsd_pairs_available_finite"]
    assert not report["acceptance"]["overall_accepted"]
    science = pd.read_csv(output / "science_metrics_comparison.csv")
    assert len(science) == 1 and not bool(science.loc[0, "available"])


@pytest.mark.parametrize("field,value", [("accelerator", "gpu"), ("workers", 2)])
def test_accelerator_and_workers_mismatches_reject_acceptance(tmp_path, field, value):
    args = _pair(tmp_path)
    baseline = json.loads(args[2].read_text())
    sensitivity = json.loads(args[3].read_text())
    baseline["run"].update({"accelerator": "cpu", "workers": 0,
                             "deterministic_policy": {"requested": True, "effective": True}})
    sensitivity["run"].update({"accelerator": "cpu", "workers": 0,
                                "deterministic_policy": {"requested": True, "effective": True}})
    sensitivity["run"][field] = value
    args[2].write_text(json.dumps(baseline)); args[3].write_text(json.dumps(sensitivity))
    output = compare_runs(*args, tmp_path / f"mismatch-{field}")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["fixed_design"][f"{field}_identical"]
    assert not report["acceptance"]["overall_accepted"]


def test_missing_execution_provenance_rejects_acceptance(tmp_path):
    args = _pair(tmp_path)
    sensitivity = json.loads(args[3].read_text())
    sensitivity["run"].pop("workers")
    args[3].write_text(json.dumps(sensitivity))
    output = compare_runs(*args, tmp_path / "missing-execution-provenance")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["fixed_design"]["workers_identical"]
    assert not report["acceptance"]["overall_accepted"]


def test_deterministic_policy_mismatch_rejects_paired_acceptance(tmp_path):
    args = _pair(tmp_path)
    for manifest in (args[2], args[3]):
        payload = json.loads(manifest.read_text())
        payload["run"].update({"accelerator": "cpu", "workers": 0,
                               "deterministic_policy": {"requested": True, "effective": True}})
        manifest.write_text(json.dumps(payload))
    sensitivity = json.loads(args[3].read_text())
    sensitivity["run"]["deterministic_policy"]["effective"] = False
    args[3].write_text(json.dumps(sensitivity))
    output = compare_runs(*args, tmp_path / "mismatch-determinism")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["fixed_design"]["deterministic_policy_identical"]
    assert not report["acceptance"]["overall_accepted"]


def test_fixed_design_mismatch_rejects_acceptance(tmp_path):
    args = _pair(tmp_path)
    mismatched = json.loads(args[3].read_text())
    mismatched["run"]["seed"] = 7
    args[3].write_text(json.dumps(mismatched))
    output = compare_runs(*args, tmp_path / "mismatch")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["acceptance"]["fixed_design_contract"]
    assert not report["fixed_design"]["seed_identical"]
    assert not report["acceptance"]["overall_accepted"]


@pytest.mark.parametrize("field,value", [
    ("distance_qc_expected_um", 99.0), ("distance_qc_relative_tolerance", 0.4),
    ("minimum_domain_assignment_coverage", 0.8), ("slide_key", "other"),
    ("group_key", "other_patient"), ("reference", "subset"), ("inference_mode", "finetuned"),
])
def test_fixed_design_distance_and_mode_mismatches_reject(tmp_path, field, value):
    args = _pair(tmp_path)
    mismatched = json.loads(args[3].read_text())
    mismatched["run"][field] = value
    args[3].write_text(json.dumps(mismatched))
    output = compare_runs(*args, tmp_path / f"mismatch-{field}")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["fixed_design"]["fixed_design_contract"]
    assert not report["acceptance"]["overall_accepted"]


def test_sensitivity_and_comparison_launchers_render_fixed_protocol(tmp_path):
    root = Path(__file__).parents[1]
    sens = root / "scripts" / "submit_novae_nominal_100um_sensitivity.sh"
    env = {**os.environ, "NOVAE_REPO_DIR": str(root),
           "NOVAE_SENSITIVITY_RUN_ROOT": str(tmp_path / "sens-run"),
           "NOVAE_SENSITIVITY_INPUT_H5AD": str(tmp_path / "source.h5ad"),
           "NOVAE_SENSITIVITY_MODEL_PATH": str(tmp_path / "model"),
           "NOVAE_SENSITIVITY_SCALE_MANIFEST": str(tmp_path / "scales.csv")}
    subprocess.run(["bash", str(sens), "--render-only"], env=env, check=True, capture_output=True, text=True)
    rendered = (tmp_path / "sens-run" / "submit_novae_nominal_100um_sensitivity.sbatch").read_text()
    assert "--coordinate-strategy visium_explicit_scale" in rendered and "--graph-radius-um" not in rendered
    assert "#SBATCH --cpus-per-task=2" in rendered and "--workers 2" in rendered
    bad = {**env, "NOVAE_RESOLUTIONS": "0.5 1.0"}
    assert subprocess.run(["bash", str(sens), "--render-only"], env=bad, capture_output=True).returncode == 2
    bad_workers = {**env, "NOVAE_WORKERS": "8"}
    assert subprocess.run(["bash", str(sens), "--render-only"], env=bad_workers, capture_output=True).returncode == 2
    bad_cpus = {**env, "NOVAE_CPUS_PER_TASK": "8"}
    assert subprocess.run(["bash", str(sens), "--render-only"], env=bad_cpus, capture_output=True).returncode == 2
    compare = root / "scripts" / "submit_novae_comparison.sh"
    cenv = {**os.environ, "NOVAE_REPO_DIR": str(root), "NOVAE_COMPARISON_RUN_ROOT": str(tmp_path / "compare-run"),
            "NOVAE_BASELINE_H5AD": str(tmp_path / "b.h5ad"), "NOVAE_SENSITIVITY_H5AD": str(tmp_path / "s.h5ad"),
            "NOVAE_COMPARISON_OUTPUT_DIR": str(tmp_path / "comparison")}
    subprocess.run(["bash", str(compare), "--render-only"], env=cenv, check=True, capture_output=True, text=True)
    compare_text = (tmp_path / "compare-run" / "submit_novae_comparison.sbatch").read_text()
    assert "novae_resolved_manifest_skin_visium_ssc.json" in compare_text
    assert "novae_resolved_manifest_skin_visium_ssc_nominal_100um_sensitivity.json" in compare_text
    unsafe = {**cenv, "NOVAE_COMPARISON_LOG_DIR": str(tmp_path / "unsafe path")}
    assert subprocess.run(["bash", str(compare), "--render-only"], env=unsafe, capture_output=True).returncode == 2


def test_paired_cpu_launcher_renders_both_arms_and_is_fail_closed(tmp_path):
    root = Path(__file__).parents[1]
    script = root / "scripts" / "submit_novae_paired_cpu_diagnostic.sh"
    env = {**os.environ, "NOVAE_REPO_DIR": str(root),
           "NOVAE_PAIRED_RUN_ROOT": str(tmp_path / "run"),
           "NOVAE_PAIRED_INPUT_H5AD": str(tmp_path / "source.h5ad"),
           "NOVAE_PAIRED_ORIGINAL_MANIFEST": str(tmp_path / "original.csv"),
           "NOVAE_PAIRED_SCALE_MANIFEST": str(tmp_path / "scales.csv"),
           "NOVAE_PAIRED_MODEL_PATH": str(tmp_path / "model")}
    subprocess.run(["bash", str(script), "--render-only"], env=env, check=True, capture_output=True, text=True)
    rendered = (tmp_path / "run" / "submit_novae_paired_cpu_diagnostic.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=1" in rendered and "#SBATCH --nodes=1" in rendered
    assert "#SBATCH --mem=96gb" in rendered and "--gres" not in rendered
    assert rendered.count("--accelerator cpu") == 2 and rendered.count("--workers 0") == 2
    assert rendered.count("--deterministic") == 2
    assert "--coordinate-strategy visium_manifest" in rendered
    assert "--physical-spot-diameter-um 55.0 --graph-radius-um 100" in rendered
    assert "--coordinate-strategy visium_explicit_scale" in rendered
    assert "--graph-radius-um" in rendered and rendered.count("--graph-radius-um") == 1
    assert "compare_novae_runs.py" in rendered
    bad = {**env, "NOVAE_WORKERS": "2"}
    assert subprocess.run(["bash", str(script), "--render-only"], env=bad, capture_output=True).returncode == 2
    conflict = {**env, "NOVAE_OUTPUT_DIR": str(tmp_path / "leak")}
    assert subprocess.run(["bash", str(script), "--render-only"], env=conflict, capture_output=True).returncode == 2
    unsafe = {**env, "NOVAE_PAIRED_LOG_DIR": str(tmp_path / "unsafe path")}
    assert subprocess.run(["bash", str(script), "--render-only"], env=unsafe, capture_output=True).returncode == 2
    existing = tmp_path / "existing-output"
    existing.mkdir()
    blocked = {**env, "NOVAE_PAIRED_ORIGINAL_OUTPUT_DIR": str(existing)}
    assert subprocess.run(["bash", str(script), "--render-only"], env=blocked, capture_output=True).returncode == 2


def test_compare_parser_and_cli_reject_protocol_overrides(tmp_path):
    parsed = build_parser().parse_args(["--baseline-h5ad", "b.h5ad", "--baseline-manifest", "bm.json",
                                        "--sensitivity-h5ad", "s.h5ad", "--sensitivity-manifest", "sm.json",
                                        "--output-dir", "out", "--minimum-coverage", "0"])
    assert parsed.minimum_coverage == 0
    args = _pair(tmp_path)
    cli_args = ["--baseline-h5ad", str(args[0]), "--baseline-manifest", str(args[2]),
                "--sensitivity-h5ad", str(args[1]), "--sensitivity-manifest", str(args[3]),
                "--output-dir", str(tmp_path / "cli-bad"), "--minimum-coverage", "0"]
    assert main(cli_args) == 2
    cli_args = ["--baseline-h5ad", str(args[0]), "--baseline-manifest", str(args[2]),
                "--sensitivity-h5ad", str(args[1]), "--sensitivity-manifest", str(args[3]),
                "--output-dir", str(tmp_path / "cli-bad-slide"), "--slide-key", "other_slide"]
    assert main(cli_args) == 2


def test_compare_protocol_rejects_coverage_and_slide_overrides(tmp_path):
    args = _pair(tmp_path)
    with pytest.raises(NovaComparisonError, match="fixed.*0.70"):
        compare_runs(*args, tmp_path / "bad-coverage", minimum_coverage=0.0)
    with pytest.raises(NovaComparisonError, match="fixed.*sample_id"):
        compare_runs(*args, tmp_path / "bad-slide", slide_key="other_slide")


def test_differing_validity_masks_are_reported_per_side_and_reject_acceptance(tmp_path):
    args = _pair(tmp_path)
    sensitivity = ad.read_h5ad(args[1])
    sensitivity.obs.loc["d", "neighborhood_valid"] = False
    sensitivity.write_h5ad(args[1])
    output = compare_runs(*args, tmp_path / "mask-mismatch")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["acceptance"]["validity_masks_exact"]
    slide_rows = pd.read_csv(output / "latent_per_slide_comparison.csv").set_index("slide")
    assert slide_rows.loc["B", "baseline_valid_rows"] == 2
    assert slide_rows.loc["B", "sensitivity_valid_rows"] == 1
    assert not report["acceptance"]["overall_accepted"]


def test_launcher_rejects_unwired_scalar_coordinate_strategies(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "submit_novae_skin_pilot.sh"
    env = {**os.environ, "NOVAE_REPO_DIR": str(Path(__file__).parents[1]), "NOVAE_RUN_ROOT": str(tmp_path / "run"),
           "NOVAE_COORDINATE_STRATEGY": "materialized_microns"}
    assert subprocess.run(["bash", str(script), "--render-only"], env=env, capture_output=True).returncode == 2


def test_comparison_rejects_graph_difference_and_cleans_failure(tmp_path):
    args = _pair(tmp_path, altered_graph=True)
    output = compare_runs(*args, tmp_path / "comparison")
    report = json.loads((output / "novae_comparison.json").read_text())
    assert not report["acceptance"]["graph_identity"]
    with pytest.raises(NovaComparisonError, match="existing"):
        compare_runs(*args, output)
