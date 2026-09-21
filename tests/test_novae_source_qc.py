from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.audit_skin_visium_source_geometry import SourceGeometryError, audit_archive, audit_source, write_audit


def _zip(path: Path, rows: str, scales: dict | None = None, member: str = "S1/spatial/tissue_positions.csv") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, rows)
        archive.writestr("S1/spatial/scalefactors_json.json", json.dumps(scales or {"spot_diameter_fullres": 10.0}))


def _rows(header: bool = True) -> str:
    values = ["a,1,0,0,0,0", "b,1,0,2,0,10", "c,1,1,-1,8,5", "d,0,1,1,8,15"]
    return ("barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n" if header else "") + "\n".join(values) + "\n"


def test_hex_lattice_edges_scales_and_legacy_format(tmp_path):
    archive = tmp_path / "S1.zip"
    _zip(archive, _rows(header=False), {"spot_diameter_fullres": 10, "tissue_hires_scalef": .2, "tissue_lowres_scalef": .02})
    result = audit_archive(archive)
    assert result["canonical_lattice_edges_all"] == 5
    assert result["lattice_connected_components_all"] == 1
    assert result["pixel_pitch_median"] == pytest.approx(10)
    assert result["current_55um_resulting_median_um"] == pytest.approx(55)
    assert result["nominal_100um_array_pitch_scale_um_per_pixel"] == pytest.approx(10)
    assert result["tissue_hires_scalef"] == pytest.approx(.2)


@pytest.mark.parametrize("rows", [
    _rows().replace("a,1,0,0,0,0", "a,1,0,0,nan,0"),
    _rows().replace("d,0,1,1,8,15", "d,0,0,0,8,15"),
    _rows().replace("d,0,1,1,8,15", "d,0,1,1,0,10"),
    "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
    "a,1,0,0,0,0\nb,1,0,1,0,10\nc,1,2,2,8,5\nd,0,4,4,8,15\n",
])
def test_source_fail_closed(tmp_path, rows):
    archive = tmp_path / "bad.zip"
    _zip(archive, rows)
    with pytest.raises(SourceGeometryError):
        audit_archive(archive)


def test_duplicate_barcode_rejected_with_precise_error(tmp_path):
    archive = tmp_path / "duplicate-barcode.zip"
    _zip(archive, _rows().replace("b,1,0,2,0,10", "a,1,0,2,0,10"))
    with pytest.raises(SourceGeometryError, match=r"duplicate barcode 'a'.*row 3.*row 2"):
        audit_archive(archive)


def test_missing_lattice_gap_is_reported_as_observed_topology(tmp_path):
    archive = tmp_path / "gap.zip"
    rows = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n" "a,1,0,0,0,0\n" "b,1,0,2,0,10\n" "c,1,0,6,0,30\n"
    _zip(archive, rows)
    result = audit_archive(archive)
    assert result["canonical_lattice_edges_all"] == 1
    assert result["lattice_zero_degree_all"] == 1
    assert result["lattice_connected_components_all"] == 2
    # This is an observed-site topology count, not a claim that a missing site was detected.


def test_selection_excludes_stereo_and_atomic_refuses_existing(tmp_path):
    _zip(tmp_path / "S1.zip", _rows())
    _zip(tmp_path / "Stereo_seq_S2.zip", _rows())
    rows = audit_source(tmp_path)
    assert [row["sample"] for row in rows] == ["S1"]
    output = write_audit(rows, tmp_path / "out", proposed_manifest=True)
    candidate = output / "skin_visium_nominal_sensitivity_candidate_scales.csv"
    assert candidate.is_file() and "sample_id,microns_per_pixel,scale_source" in candidate.read_text()
    with pytest.raises(SourceGeometryError, match="existing"):
        write_audit(rows, output)


def test_h5ad_launcher_render_only(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "submit_novae_h5ad_qc.sh"
    run_root = tmp_path / "run"
    env = {
        "NOVAE_REPO_DIR": str(Path(__file__).parents[1]), "NOVAE_RUN_ROOT": str(run_root),
        "NOVAE_SOURCE_H5AD": str(tmp_path / "source.h5ad"), "NOVAE_ANNOTATED_H5AD": str(tmp_path / "annotated.h5ad"),
        "NOVAE_H5AD_QC_OUTPUT_DIR": str(tmp_path / "out"), "NOVAE_H5AD_QC_LOG_DIR": str(tmp_path / "logs"),
        "NOVAE_H5AD_QC_JOB_SCRIPT": str(run_root / "job.sbatch"),
    }
    result = subprocess.run(["bash", str(script), "--render-only"], env={**os.environ, **env}, check=True, capture_output=True, text=True)
    rendered = (run_root / "job.sbatch").read_text()
    assert "#SBATCH --gres" not in rendered
    assert "#SBATCH --cpus-per-task=1" in rendered and "--mem=64gb" in rendered
    assert "audit_novae_h5ad_qc.py" in rendered
    assert result.returncode == 0
    override = {**os.environ, **env, "NOVAE_H5AD_QC_DOMAIN_COLUMNS": "novae_domains_res0.5,novae_domains_res1.0"}
    subprocess.run(["bash", str(script), "--render-only"], env=override, check=True, capture_output=True, text=True)
    assert "DOMAIN_ARGS=(--domain-columns" in (run_root / "job.sbatch").read_text()
    unsafe = {**os.environ, **env, "NOVAE_ACCOUNT": "bad;account"}
    assert subprocess.run(["bash", str(script), "--render-only"], env=unsafe, capture_output=True).returncode == 2


@pytest.mark.parametrize("bad", [[[-1, 0]], [[0.5, 0]], [[float("nan"), 0]]])
def test_raw_count_contract_rejects_negative_fractional_nonfinite(bad):
    np = pytest.importorskip("numpy")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    from scripts.audit_novae_h5ad_qc import H5ADQCCError, validate_raw_counts
    matrix = scipy_sparse.csr_matrix(np.asarray(bad, dtype=float))
    with pytest.raises(H5ADQCCError, match="raw counts"):
        validate_raw_counts(matrix)


def _tiny_h5ad_pair(tmp_path):
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    ids = [f"s{i}" for i in range(4)]
    source = ad.AnnData(scipy_sparse.csr_matrix([[0, 0], [1, 0], [0, 0], [2, 0]]))
    source.obs_names = ids
    source.obs["sample_id"] = ["A", "A", "B", "B"]
    annotated = source.copy()
    annotated.obs["neighborhood_valid"] = [True, False, True, False]
    annotated.obs["novae_domains_res0.5"] = ["D1", np.nan, "D2", np.nan]
    annotated.obsm["spatial"] = np.arange(8, dtype=float).reshape(4, 2)
    graph = scipy_sparse.csr_matrix((np.ones(4), ([0, 1, 2, 3], [1, 0, 3, 2])), shape=(4, 4))
    annotated.obsp["spatial_connectivities"] = graph
    source_path, annotated_path = tmp_path / "source.h5ad", tmp_path / "annotated.h5ad"
    source.write_h5ad(source_path)
    annotated.write_h5ad(annotated_path)
    return source, annotated, source_path, annotated_path


def test_h5ad_cross_tabs_preserve_na_and_misalignment(tmp_path):
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    from scripts.audit_novae_h5ad_qc import H5ADQCCError, run_audit
    ids = [f"s{i}" for i in range(6)]
    source = ad.AnnData(scipy_sparse.csr_matrix([[0, 0], [1, 0], [0, 0], [2, 0], [1, 1], [0, 0]]))
    source.obs_names = ids
    source.obs["sample_id"] = ["A", "A", "A", "B", "B", "B"]
    annotated = source.copy()
    annotated.obs["neighborhood_valid"] = [True, False, True, True, False, True]
    annotated.obs["novae_domains_res0.5"] = ["D1", np.nan, "D1", "D2", np.nan, "D2"]
    annotated.obsm["spatial"] = np.arange(12, dtype=float).reshape(6, 2)
    graph = scipy_sparse.csr_matrix((np.ones(6), ([0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3])), shape=(6, 6))
    annotated.obsp["spatial_connectivities"] = graph
    source_path, annotated_path = tmp_path / "source.h5ad", tmp_path / "annotated.h5ad"
    source.write_h5ad(source_path)
    annotated.write_h5ad(annotated_path)
    output = run_audit(source_path, annotated_path, tmp_path / "qc")
    import pandas as pd
    cross = pd.read_csv(output / "zero_count_x_zero_degree_x_validity.csv")
    invalid = pd.read_csv(output / "invalid_neighborhoods.csv")
    assert len(cross) == 8 and int(cross["count"].sum()) == 6
    assert len(invalid) == 2
    assert pd.isna(invalid.loc[0, "novae_domains_res0.5"])
    altered = annotated.copy()
    altered.obs_names = list(reversed(ids))
    altered.write_h5ad(tmp_path / "misaligned.h5ad")
    with pytest.raises(H5ADQCCError, match="aligned"):
        run_audit(source_path, tmp_path / "misaligned.h5ad", tmp_path / "qc2")
    with pytest.raises(H5ADQCCError, match="existing"):
        run_audit(source_path, annotated_path, output)


def test_h5ad_missing_slide_and_domain_consistency_fail_closed(tmp_path):
    _source, annotated, source_path, _annotated_path = _tiny_h5ad_pair(tmp_path)
    from scripts.audit_novae_h5ad_qc import H5ADQCCError, run_audit
    missing = annotated.copy()
    missing.obs.iloc[0, missing.obs.columns.get_loc("sample_id")] = None
    missing.write_h5ad(tmp_path / "missing-slide.h5ad")
    with pytest.raises(H5ADQCCError, match="missing values"):
        run_audit(source_path, tmp_path / "missing-slide.h5ad", tmp_path / "missing-out")
    inconsistent = annotated.copy()
    inconsistent.obs.iloc[0, inconsistent.obs.columns.get_loc("novae_domains_res0.5")] = float("nan")
    inconsistent.write_h5ad(tmp_path / "inconsistent.h5ad")
    with pytest.raises(H5ADQCCError, match="violates neighborhood_valid"):
        run_audit(source_path, tmp_path / "inconsistent.h5ad", tmp_path / "inconsistent-out")


def test_h5ad_transaction_cleanup_on_late_write_failure(tmp_path, monkeypatch):
    _source, _annotated, source_path, annotated_path = _tiny_h5ad_pair(tmp_path)
    import scripts.audit_novae_h5ad_qc as qc
    def fail_json(*_args, **_kwargs):
        raise OSError("forced late write failure")
    monkeypatch.setattr(qc, "_atomic_json", fail_json)
    output = tmp_path / "late-failure"
    with pytest.raises(OSError, match="forced late"):
        qc.run_audit(source_path, annotated_path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.partial"))
