"""Tests for the FOV MLP input builder, focused on the label-collapse option.

`--label-map` exists so the two-class IBD task (HC vs IBD) can be evaluated from
the same features and patient grouping as the three-class task (HC/UC/CD). It
produces numbers intended for publication, so the invariants it must hold are
worth pinning down: only the target changes, never the features or the groups.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "pipeline_assets" / "IBD_Build_FOV_MLP_Inputs.py"

pytest.importorskip("anndata", reason="builder reads h5ad")
pytest.importorskip("pyarrow", reason="builder writes parquet")

DISEASE_BY_PATIENT = {
    "P1": "HC",
    "P2": "HC",
    "P3": "UC",
    "P4": "UC",
    "P5": "CD",
    "P6": "CD",
}


def _make_inputs(root: Path) -> tuple[Path, Path]:
    """Write a minimal cosmx_with_nmf.h5ad plus the post-NMF feature tables."""
    import anndata as ad

    rng = np.random.default_rng(0)
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)

    rows = []
    for patient, disease in DISEASE_BY_PATIENT.items():
        for fov in range(3):
            for _cell in range(5):
                rows.append((patient, disease, fov, str(rng.integers(0, 4))))
    obs = pd.DataFrame(rows, columns=["patient", "disease_state", "fov", "NMF_factor"])
    obs.index = [f"cell{i}" for i in range(len(obs))]

    adata = ad.AnnData(X=rng.normal(size=(len(obs), 3)).astype("float32"), obs=obs)
    h5ad_path = source / "cosmx_with_nmf.h5ad"
    adata.write_h5ad(h5ad_path)

    fov_keys = sorted({f"{p}_{f}" for p in DISEASE_BY_PATIENT for f in range(3)})
    index = pd.Index(fov_keys, name="fov_key")

    # enrichment_<i>-<j>; the builder keeps only pairs with i < j.
    enrichment = pd.DataFrame(
        rng.normal(size=(len(index), 3)),
        index=index,
        columns=["enrichment_0-1", "enrichment_0-2", "enrichment_1-2"],
    )
    enrichment.to_parquet(source / "enrichment_features_fov.parquet")

    niche_gene = pd.DataFrame(
        rng.normal(size=(len(index), 4)),
        index=index,
        columns=[f"niche_0_gene_G{i}" for i in range(4)],
    )
    niche_gene.to_parquet(source / "niche_gene_features_fov.parquet")

    pd.DataFrame(
        {"group": ["g"] * 4, "feature": list(niche_gene.columns)}
    ).to_csv(source / "niche_gene_ranked_features.csv", index=False)

    return source, h5ad_path


def _run_builder(source: Path, h5ad: Path, dest: Path, label_map: str | None = None):
    cmd = [
        sys.executable,
        str(BUILDER),
        "--output-dir",
        str(source),
        "--cosmx-with-nmf",
        str(h5ad),
        "--dest-dir",
        str(dest),
        "--niche-gene-count",
        "4",
    ]
    if label_map is not None:
        cmd += ["--label-map", label_map]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def test_builder_without_label_map_keeps_three_classes(tmp_path: Path) -> None:
    source, h5ad = _make_inputs(tmp_path)
    dest = tmp_path / "out3"
    result = _run_builder(source, h5ad, dest)
    assert result.returncode == 0, result.stderr[-3000:]

    targets = pd.read_parquet(dest / "targets_y.parquet").squeeze()
    assert set(targets.unique()) == {"HC", "UC", "CD"}


def test_label_map_collapses_target_only(tmp_path: Path) -> None:
    """The two-class task must differ from the three-class task in the target alone."""
    source, h5ad = _make_inputs(tmp_path)
    dest3, dest2 = tmp_path / "out3", tmp_path / "out2"

    assert _run_builder(source, h5ad, dest3).returncode == 0
    result = _run_builder(source, h5ad, dest2, "UC=IBD,CD=IBD")
    assert result.returncode == 0, result.stderr[-3000:]

    y3 = pd.read_parquet(dest3 / "targets_y.parquet").squeeze()
    y2 = pd.read_parquet(dest2 / "targets_y.parquet").squeeze()
    assert set(y2.unique()) == {"HC", "IBD"}

    # HC stays HC; UC and CD both become IBD; row order is preserved.
    expected = y3.astype(str).map({"HC": "HC", "UC": "IBD", "CD": "IBD"})
    pd.testing.assert_series_equal(
        y2.astype(str).reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )

    # Features and patient groups must be byte-identical between the two tasks.
    x3 = pd.read_parquet(dest3 / "combined_features_filtered.parquet")
    x2 = pd.read_parquet(dest2 / "combined_features_filtered.parquet")
    pd.testing.assert_frame_equal(x3, x2)

    g3 = pd.read_parquet(dest3 / "groups.parquet")
    g2 = pd.read_parquet(dest2 / "groups.parquet")
    pd.testing.assert_frame_equal(g3, g2)


def test_label_map_rejects_unknown_label(tmp_path: Path) -> None:
    """A typo in the map must fail loudly rather than silently doing nothing."""
    source, h5ad = _make_inputs(tmp_path)
    result = _run_builder(source, h5ad, tmp_path / "bad", "TYPO=IBD")
    assert result.returncode != 0
    assert "not present in disease_state" in (result.stdout + result.stderr)


def test_label_map_rejects_malformed_entry(tmp_path: Path) -> None:
    source, h5ad = _make_inputs(tmp_path)
    result = _run_builder(source, h5ad, tmp_path / "bad", "UC->IBD")
    assert result.returncode != 0
    assert "FROM=TO" in (result.stdout + result.stderr)
