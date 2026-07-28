"""Executable guards against evaluation leakage.

The defect that motivated these tests is recorded in
`docs/MLP_EVALUATION_AND_LEAKAGE.md`: an evaluation that selects hyperparameters
on the full dataset and then reports performance on the folds that selected them
returns a selection maximum, not a held-out estimate.

The invariant being protected is simple and cheap to check:

    On data where the label is independent of every feature, any honest
    evaluation must land at chance. Anything above chance is leakage.

`test_leakage_safe_evaluation_is_chance_on_null_data` applies that to the real
pipeline script. `test_search_then_report_on_same_folds_inflates_score`
demonstrates the defect itself, so the bug class stays visible and cannot
quietly return.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synthetic_cohort import majority_baseline, make_cohort

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAKAGE_SAFE_SCRIPT = REPO_ROOT / "pipeline_assets" / "IBD_MLP_LeakageSafe.py"

pytest.importorskip("sklearn", reason="pipeline evaluation tests need scikit-learn")
pytest.importorskip("pyarrow", reason="pipeline stages exchange parquet")

FIXED_PARAMS = {
    "hidden_layer_sizes": [16],
    "activation": "relu",
    "alpha": 1e-3,
    "learning_rate_init": 1e-2,
    "batch_size": 16,
    "backend": "sklearn",
    "device": "auto",
    "max_epochs": 60,
    "patience": 10,
}


def _run_leakage_safe(cohort_dir: Path, out_dir: Path, mode: str, extra_env: dict | None = None) -> str:
    env = dict(os.environ)
    env.update(
        {
            "NICHERUNNER_OUTPUT_DIR": str(cohort_dir),
            "NICHERUNNER_SOURCE_OUTPUT_DIR": str(cohort_dir),
            "NICHERUNNER_MLP_OUTPUT_DIR": str(out_dir),
            "NICHERUNNER_MLP_UNIT": "fov",
            "NICHERUNNER_MLP_MODE": mode,
            "NICHERUNNER_MLP_BACKEND": "sklearn",
            "NICHERUNNER_MLP_GRID_PROFILE": "compact",
            "NICHERUNNER_MLP_MAX_EPOCHS": "60",
            "NICHERUNNER_SKIP_SHAP": "1",
            "NICHERUNNER_TOP_ENRICHMENT_FEATURES": "4",
            "NICHERUNNER_TOP_NICHE_GENE_FEATURES": "5",
        }
    )
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, str(LEAKAGE_SAFE_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, f"script failed:\n{result.stderr[-4000:]}"
    return (out_dir / "mlp_results.txt").read_text(encoding="utf-8")


def _pooled_accuracy(report_text: str) -> float:
    """Read the pooled accuracy out of the overall classification report.

    Pooled accuracy is the statistically meaningful summary here; the per-fold
    mean printed above it degenerates under leave-one-group-out (see
    docs/MLP_EVALUATION_AND_LEAKAGE.md section 4.2).
    """
    match = re.search(r"^\s*accuracy\s+([0-9.]+)\s+\d+\s*$", report_text, re.M)
    assert match, f"could not parse pooled accuracy from:\n{report_text[-2000:]}"
    return float(match.group(1))


def test_leakage_safe_evaluation_is_chance_on_null_data(tmp_path: Path) -> None:
    """Labels independent of features must evaluate at chance.

    Exercises the real outer loop: patient-grouped LeaveOneGroupOut, in-fold
    mutual-information feature selection, and in-fold scaling. Uses
    evaluate_fixed so the inner grid search does not dominate runtime; the
    nested_cv equivalent is covered by the slow test below.
    """
    cohort = tmp_path / "cohort"
    make_cohort(cohort, n_patients=12, fovs_per_patient=8, signal=0.0, seed=11)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "fixed_params.json").write_text(json.dumps(FIXED_PARAMS), encoding="utf-8")

    report = _run_leakage_safe(cohort, out_dir, "evaluate_fixed")
    accuracy = _pooled_accuracy(report)

    # Chance is 0.5. The bound is loose because leave-one-patient-out on 12
    # patients is noisy; it is still far below what leakage produces.
    assert accuracy < 0.70, (
        f"pooled accuracy {accuracy:.3f} on label-independent features. "
        "An honest evaluation cannot beat chance here - this indicates leakage."
    )


def test_leakage_safe_evaluation_detects_real_signal(tmp_path: Path) -> None:
    """Positive control: the harness must not be blind to genuine structure.

    Without this, the null test above could pass simply because the evaluation
    is broken and predicts one class forever.
    """
    cohort = tmp_path / "cohort"
    info = make_cohort(cohort, n_patients=12, fovs_per_patient=8, signal=2.5, seed=11)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "fixed_params.json").write_text(json.dumps(FIXED_PARAMS), encoding="utf-8")

    report = _run_leakage_safe(cohort, out_dir, "evaluate_fixed")
    accuracy = _pooled_accuracy(report)

    baseline = majority_baseline(info["labels"])
    assert accuracy > baseline, (
        f"pooled accuracy {accuracy:.3f} did not beat the majority baseline "
        f"{baseline:.3f} on data with a strong planted signal."
    )


@pytest.mark.slow
def test_nested_cv_is_chance_on_null_data(tmp_path: Path) -> None:
    """Same invariant for the full nested_cv path, including inner tuning.

    Slower because the inner grouped grid search runs per outer fold. Marked
    slow and excluded from the default run; execute with `-m slow`.
    """
    cohort = tmp_path / "cohort"
    make_cohort(cohort, n_patients=8, fovs_per_patient=6, signal=0.0, seed=5)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    report = _run_leakage_safe(
        cohort, out_dir, "nested_cv", extra_env={"NICHERUNNER_MLP_MAX_EPOCHS": "30"}
    )
    accuracy = _pooled_accuracy(report)
    assert accuracy < 0.75, (
        f"nested_cv pooled accuracy {accuracy:.3f} on label-independent features."
    )


def test_search_then_report_on_same_folds_inflates_score() -> None:
    """Demonstrate the defect the discontinued MLP scripts contain.

    `IBD_MLP_44Features.py` calls RandomizedSearchCV over the full dataset with a
    given splitter and then reports a "final cross-validation" using that same
    splitter, so `best_score_` and the reported metric are the same quantity.

    The defining property is relative, not absolute: what gets reported is the
    *maximum* over the candidate distribution evaluated on those folds, so it
    exceeds the typical candidate by an amount that grows with the number of
    candidates. The real script used n_iter=30000. Absolute scores on noise
    depend on the draw, which is why this asserts on the gap rather than on a
    fixed threshold.

    Uses StratifiedGroupKFold(3) to match `cv_mode: sgkf3`, the setting that
    produced the skin manuscript output.

    Kept as an executable record of the bug class: if someone reintroduces the
    pattern, this test explains precisely why it is wrong.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_patients, per_patient, n_features = 18, 8, 50
    inflations = []

    for seed in range(5):
        rng = np.random.default_rng(seed)
        groups = np.repeat(np.arange(n_patients), per_patient)
        # Disease state is patient-level, so labels are assigned per patient.
        patient_labels = rng.permutation([0, 1] * (n_patients // 2))
        y = np.repeat(patient_labels, per_patient)
        X = rng.normal(size=(n_patients * per_patient, n_features))  # pure noise

        pipe = Pipeline(
            [
                ("sel", SelectKBest(f_classif)),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        grid = {"sel__k": [1, 2, 3, 5, 8, 12, 20, 30], "clf__C": np.logspace(-3, 3, 25)}
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted")
        search.fit(X, y, groups=groups)

        scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
        reported = float(search.best_score_)  # what the discontinued script prints
        typical = float(np.median(scores))
        inflations.append(reported - typical)

        assert reported > typical, (
            f"seed {seed}: reporting the search maximum ({reported:.3f}) should exceed "
            f"the median candidate ({typical:.3f}) on label-independent data"
        )

    mean_inflation = float(np.mean(inflations))
    assert mean_inflation > 0.02, (
        f"mean inflation {mean_inflation:.3f} over {len(inflations)} draws is smaller "
        "than expected; the demonstration may no longer reflect the defect"
    )
