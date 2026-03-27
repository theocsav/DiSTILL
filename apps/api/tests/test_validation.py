from pathlib import Path

from app.validation import validate_config


def _base_config() -> dict:
    return {
        "run_name": "validation-demo",
        "mode": "fixed_k",
        "n_components": 4,
        "stages": ["post_nmf"],
        "cosmx_h5ad_path": "/tmp/cosmx.h5ad",
        "reference_h5ad_path": "/tmp/reference.h5ad",
        "cell_metadata_path": "/tmp/metadata.csv",
        "check_join_keys": False,
    }


def test_post_nmf_standalone_requires_nmf_artifact_when_cell2loc_not_selected(monkeypatch) -> None:
    monkeypatch.setattr("app.validation.QUEUE_POLLER_TOKEN", "")
    errors, warnings, checks = validate_config(_base_config(), check_paths=False)

    assert any("post_nmf without cell2loc_nmf requires cosmx_with_nmf_path" in error for error in errors)
    assert warnings == []
    assert "stage_data_contract" not in checks


def test_post_nmf_with_cell2loc_does_not_require_preexisting_nmf_artifact() -> None:
    config = _base_config()
    config["stages"] = ["cell2loc_nmf", "post_nmf"]

    errors, _warnings, _checks = validate_config(config, check_paths=False)

    assert not any("cosmx_with_nmf_path" in error for error in errors)


def test_stage_data_contract_reports_missing_coords_and_morphology(monkeypatch) -> None:
    config = _base_config()
    config["cosmx_with_nmf_path"] = "/tmp/cosmx_with_nmf.h5ad"

    def fake_obs(_path: Path):
        class _ObsFrame:
            columns = ["patient", "cell_id", "fov"]

        return _ObsFrame()

    def fake_header(_path: Path):
        return ["fov", "cell_ID"]

    monkeypatch.setattr("app.validation._read_h5ad_obs", fake_obs)
    monkeypatch.setattr("app.validation._read_metadata_header", fake_header)

    errors, _warnings, _checks = validate_config(config, check_paths=True)

    assert any("post_nmf requires spatial coordinates" in error for error in errors)
    assert any("post_nmf requires morphology" in error for error in errors)


def test_stage_data_contract_uses_dataset_manifest_without_reading_files(monkeypatch) -> None:
    config = _base_config()
    config["dataset_id"] = "dataset-1"
    config["check_join_keys"] = False

    monkeypatch.setattr(
        "app.validation.get_dataset",
        lambda dataset_id: {
            "id": dataset_id,
            "schema_manifest": {
                "obs_keys": ["patient", "cell_id", "fov", "NMF_factor"],
                "has_spatial_coordinates": False,
                "has_morphology": False,
                "has_nmf_labels": True,
                "has_raw_counts": True,
            },
            "metadata_columns": ["fov", "cell_ID", "CenterX_global_px", "CenterY_global_px", "Area"],
            "metadata_manifest": {
                "has_spatial_coordinates": True,
                "has_morphology": True,
                "has_join_keys": True,
            },
        },
    )

    def fail_read(*_args, **_kwargs):
        raise AssertionError("should not read files when dataset manifests are available")

    monkeypatch.setattr("app.validation._read_h5ad_obs", fail_read)
    monkeypatch.setattr("app.validation._read_metadata_header", fail_read)
    monkeypatch.setattr("app.validation.QUEUE_POLLER_TOKEN", "")

    errors, warnings, checks = validate_config(config, check_paths=False)

    assert not any("spatial coordinates" in error for error in errors)
    assert not any("morphology" in error for error in errors)
    assert warnings == []
    assert checks["stage_data_contract"]["metadata_columns"] == ["fov", "cell_ID", "CenterX_global_px", "CenterY_global_px", "Area"]


def test_validation_surfaces_external_poller_compute_access_as_unknown(tmp_path: Path, monkeypatch) -> None:
    config = {
        "run_name": "validation-demo",
        "mode": "fixed_k",
        "n_components": 4,
        "stages": ["cell2loc_nmf"],
        "cosmx_h5ad_path": str(tmp_path / "cosmx.h5ad"),
        "reference_h5ad_path": str(tmp_path / "reference.h5ad"),
        "cell_metadata_path": str(tmp_path / "metadata.csv"),
        "check_join_keys": False,
    }
    for name in ("cosmx.h5ad", "reference.h5ad", "metadata.csv"):
        (tmp_path / name).write_text("stub", encoding="utf-8")

    monkeypatch.setattr("app.validation.ARTIFACT_ROOTS", [tmp_path])
    monkeypatch.setattr("app.validation.QUEUE_POLLER_TOKEN", "poller-token")
    monkeypatch.setattr("app.validation.remote_path_exists", lambda _path: True)
    monkeypatch.setattr("app.validation.os.access", lambda _path, _mode: True)

    errors, warnings, checks = validate_config(config, check_paths=True)

    assert not errors
    assert any("do not confirm HPG compute readability" in item for item in warnings)
    assert checks["compute_access"]["mode"] == "external_poller"
    assert checks["compute_access"]["cosmx_h5ad_path"] == "unknown_external_poller"
