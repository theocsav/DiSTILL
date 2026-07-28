"""Synthetic FOV cohorts for evaluation-methodology tests.

The layout mirrors what the post-NMF stage writes and what
`pipeline_assets/IBD_MLP_LeakageSafe.py` reads in `mlp_unit=fov` mode:

    <dir>/combined_features_filtered.parquet   nmf_prop_* columns
    <dir>/targets_y.parquet                    disease label per FOV
    <dir>/groups.parquet                       patient per FOV
    <dir>/enrichment_features_fov.parquet
    <dir>/niche_gene_features_fov.parquet

Disease state is a patient-level attribute, so labels are assigned per patient
and every FOV of that patient inherits it. That matters: permuting labels at FOV
level would break the patient/label correspondence and make the null test
measure the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("healthy", "systemic_sclerosis")


def make_cohort(
    directory: Path,
    *,
    n_patients: int = 12,
    fovs_per_patient: int = 8,
    n_nmf: int = 6,
    n_enrichment: int = 8,
    n_niche_gene: int = 10,
    signal: float = 0.0,
    seed: int = 0,
) -> dict:
    """Write a synthetic cohort to `directory`.

    `signal=0.0` produces pure noise features: label is independent of every
    feature, so any evaluation that reports above-chance performance on this
    cohort is leaking. `signal>0` shifts feature means by class, giving a
    positive control that the harness can detect real structure.
    """
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)

    patients = [f"P{i:02d}" for i in range(n_patients)]
    # Balanced patient-level labels, shuffled.
    patient_labels = [LABELS[i % 2] for i in range(n_patients)]
    rng.shuffle(patient_labels)
    patient_label = dict(zip(patients, patient_labels, strict=True))

    fov_ids, fov_patient, fov_label = [], [], []
    for patient in patients:
        for j in range(fovs_per_patient):
            fov_ids.append(f"{patient}_{j}")
            fov_patient.append(patient)
            fov_label.append(patient_label[patient])

    index = pd.Index(fov_ids, name="fov_key")
    n_rows = len(fov_ids)
    is_positive = np.array([lbl == LABELS[1] for lbl in fov_label], dtype=float)

    def block(n_cols: int, prefix: str) -> pd.DataFrame:
        data = rng.normal(size=(n_rows, n_cols))
        if signal:
            # Shift only the first few columns so selection has to find them.
            for c in range(min(3, n_cols)):
                data[:, c] += signal * is_positive
        return pd.DataFrame(
            data, index=index, columns=[f"{prefix}{c}" for c in range(n_cols)]
        )

    nmf = block(n_nmf, "nmf_prop_")
    enrichment = block(n_enrichment, "enrichment_")
    niche_gene = block(n_niche_gene, "niche_")

    nmf.to_parquet(directory / "combined_features_filtered.parquet")
    enrichment.to_parquet(directory / "enrichment_features_fov.parquet")
    niche_gene.to_parquet(directory / "niche_gene_features_fov.parquet")
    pd.Series(fov_label, index=index, name="y").to_frame().to_parquet(
        directory / "targets_y.parquet"
    )
    pd.Series(fov_patient, index=index, name="group").to_frame().to_parquet(
        directory / "groups.parquet"
    )

    return {
        "directory": directory,
        "n_rows": n_rows,
        "patients": patients,
        "patient_label": patient_label,
        "labels": fov_label,
        "groups": fov_patient,
        "X": pd.concat([nmf, enrichment, niche_gene], axis=1),
        "y": pd.Series(fov_label, index=index),
        "groups_series": pd.Series(fov_patient, index=index),
    }


def majority_baseline(labels) -> float:
    counts = pd.Series(list(labels)).value_counts()
    return float(counts.iloc[0] / counts.sum())
