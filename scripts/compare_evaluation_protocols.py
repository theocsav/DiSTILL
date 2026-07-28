#!/usr/bin/env python3
"""Measure how much the discontinued evaluation protocol inflates a result.

Runs two protocols over the *same* feature matrix, targets, and patient groups,
so the only variable is how hyperparameters are selected:

  leaky   select hyperparameters with a search over the full dataset using a CV
          splitter, then report that search's best score, and separately re-run
          the winning configuration over the same folds. This is what
          pipeline_assets/IBD_MLP_44Features.py does; the two numbers it yields
          are the same quantity, which is the defect.

  honest  nested: for each outer fold, select hyperparameters using only the
          training split, then predict the held-out fold once. Pooled
          predictions give the reported metrics.

Both are compared against the majority-class baseline, which is the number any
classifier has to beat to be interesting at all.

Usage:

    python scripts/compare_evaluation_protocols.py \\
        --features <combined_features_filtered.parquet> \\
        --targets  <targets_y.parquet> \\
        --groups   <groups.parquet> \\
        --outdir   <output directory> \\
        --outer-cv logo

See docs/MLP_EVALUATION_AND_LEAKAGE.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# A grid search deliberately visits configurations that will not converge at the
# given max_iter. One warning per fit buries the report and bloats SLURM logs.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Mirrors the architecture family in IBD_MLP_44Features.py, reduced to a grid
# small enough to nest. The defect being measured is structural, not a property
# of any particular grid size, though the inflation grows with candidate count.
PARAM_GRIDS = {
    "full": {
        "mlp__hidden_layer_sizes": [(32, 16, 8), (44, 22), (50, 25, 12), (64, 32), (25,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "mlp__batch_size": [4, 8, 16],
    },
    # Nesting multiplies cost by the number of inner folds, so a smaller grid
    # keeps the honest arm tractable. Inflation shrinks with fewer candidates,
    # making this the conservative choice: it understates the leak rather than
    # overstating it.
    "compact": {
        "mlp__hidden_layer_sizes": [(32, 16), (50, 25, 12), (25,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-3, 1e-2],
        "mlp__batch_size": [8, 16],
    },
}


def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, index_col=0)


def _squeeze(frame: pd.DataFrame) -> pd.Series:
    series = frame.squeeze()
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series


def _make_pipeline(max_iter: int, seed: int) -> Pipeline:
    # Scaler inside the pipeline so it is fit on training data only, in every fold.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    solver="adam",
                    learning_rate="adaptive",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def _outer_splitter(kind: str, n_splits: int, seed: int):
    if kind == "logo":
        return LeaveOneGroupOut()
    if kind == "sgkf":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"unknown cv kind: {kind}")


def _majority_baseline(y: pd.Series) -> tuple[str, float]:
    counts = y.astype(str).value_counts()
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum())


def run_leaky(X, y, groups, splitter, grid, max_iter, seed, jobs):
    """Reproduce the discontinued pattern: search and report on the same folds."""
    search = GridSearchCV(
        _make_pipeline(max_iter, seed),
        grid,
        cv=splitter,
        scoring="f1_weighted",
        n_jobs=jobs,
    )
    search.fit(X, y, groups=groups)

    # The discontinued script then re-runs the winner over the same splitter and
    # reports the MEAN OF PER-FOLD scores (np.mean(fold_f1_scores)), which is the
    # same aggregation GridSearchCV uses for best_score_. Matching it here is what
    # makes the two numbers directly comparable and shows they are one quantity.
    best = search.best_estimator_
    y_true, y_pred, fold_f1 = [], [], []
    for train_idx, test_idx in splitter.split(X, y, groups):
        best.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = best.predict(X.iloc[test_idx])
        fold_true = y.iloc[test_idx]
        fold_f1.append(float(f1_score(fold_true, fold_pred, average="weighted", zero_division=0)))
        y_pred.extend(fold_pred.tolist())
        y_true.extend(fold_true.tolist())

    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    return {
        "reported_best_score": float(search.best_score_),
        "median_candidate_score": float(np.median(scores)),
        "n_candidates": int(scores.size),
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        # Mean of per-fold scores: what the discontinued script prints as its
        # "final cross-validation" result.
        "rerun_mean_fold_weighted_f1": float(np.mean(fold_f1)),
        "rerun_fold_weighted_f1": fold_f1,
        # Pooled, for comparison with the honest arm on equal terms.
        "rerun_pooled_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "rerun_pooled_accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "y_true": [str(v) for v in y_true],
        "y_pred": [str(v) for v in y_pred],
    }


def run_honest(X, y, groups, splitter, grid, max_iter, seed, jobs):
    """Nested: hyperparameters chosen inside the training split only."""
    y_true, y_pred, fold_rows = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_tr, y_tr, g_tr = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
        inner = LeaveOneGroupOut() if g_tr.nunique() > 2 else StratifiedGroupKFold(n_splits=2)
        search = GridSearchCV(
            _make_pipeline(max_iter, seed),
            grid,
            cv=inner,
            scoring="f1_weighted",
            n_jobs=jobs,
        )
        search.fit(X_tr, y_tr, groups=g_tr)
        preds = search.best_estimator_.predict(X.iloc[test_idx])
        y_pred.extend(preds.tolist())
        y_true.extend(y.iloc[test_idx].tolist())
        fold_rows.append(
            {
                "fold": fold,
                "test_groups": sorted(groups.iloc[test_idx].astype(str).unique().tolist()),
                "n_test": int(len(test_idx)),
                "best_params": {k: str(v) for k, v in search.best_params_.items()},
            }
        )
    return {
        "pooled_accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "pooled_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "pooled_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "folds": fold_rows,
        "y_true": [str(v) for v in y_true],
        "y_pred": [str(v) for v in y_pred],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--outer-cv", choices=["logo", "sgkf"], default="logo")
    parser.add_argument("--n-splits", type=int, default=3, help="folds when --outer-cv sgkf")
    parser.add_argument("--grid", choices=sorted(PARAM_GRIDS), default="compact")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--label-map", default="", help="comma-separated FROM=TO, e.g. 'UC=IBD,CD=IBD'")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = _load(args.features)
    y = _squeeze(_load(args.targets)).astype(str)
    groups = _squeeze(_load(args.groups)).astype(str)

    common = X.index.intersection(y.index).intersection(groups.index)
    if len(common) != len(X):
        print(f"[warn] aligning on {len(common)} shared rows (features had {len(X)})")
    X, y, groups = X.loc[common], y.loc[common], groups.loc[common]
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    if args.label_map:
        mapping = {}
        for pair in args.label_map.split(","):
            if not pair.strip():
                continue
            src, dst = pair.split("=", 1)
            mapping[src.strip()] = dst.strip()
        missing = sorted(set(mapping) - set(y.unique()))
        if missing:
            raise SystemExit(f"--label-map labels not present: {missing}; present: {sorted(y.unique())}")
        y = y.map(lambda v: mapping.get(v, v))

    splitter = _outer_splitter(args.outer_cv, args.n_splits, args.seed)
    grid = PARAM_GRIDS[args.grid]
    n_grid = len(list(product(*grid.values())))

    lines = []
    def out(text=""):
        print(text)
        lines.append(text)

    out("=== Evaluation protocol comparison ===")
    out(f"rows: {len(X)}  features: {X.shape[1]}  groups: {groups.nunique()}")
    out(f"class balance: {dict(y.value_counts())}")
    out(f"outer CV: {args.outer_cv}  grid: {args.grid} ({n_grid} candidates)")
    baseline_label, baseline = _majority_baseline(y)
    out(f"majority-class baseline: {baseline:.3f} (always predict {baseline_label!r})")
    out()

    honest = run_honest(X, y, groups, splitter, grid, args.max_iter, args.seed, args.jobs)
    leaky = run_leaky(X, y, groups, splitter, grid, args.max_iter, args.seed, args.jobs)

    out("--- HONEST (nested; selection inside training split only) ---")
    out(f"pooled accuracy          : {honest['pooled_accuracy']:.3f}")
    out(f"pooled weighted F1       : {honest['pooled_weighted_f1']:.3f}")
    out(f"pooled macro F1          : {honest['pooled_macro_f1']:.3f}")
    out(f"pooled balanced accuracy : {honest['pooled_balanced_accuracy']:.3f}")
    out()
    out(classification_report(honest["y_true"], honest["y_pred"], zero_division=0))
    out("confusion matrix:")
    labels = sorted(set(honest["y_true"]))
    out(str(pd.DataFrame(
        confusion_matrix(honest["y_true"], honest["y_pred"], labels=labels),
        index=labels, columns=labels,
    )))
    out()

    out("--- LEAKY (search and report on the same folds) ---")
    out(f"search best score          : {leaky['reported_best_score']:.3f}   <- 'Best F1-Score' line")
    out(f"re-run, mean of folds      : {leaky['rerun_mean_fold_weighted_f1']:.3f}   <- 'Mean F1-Score' line, same quantity")
    out(f"re-run, pooled             : {leaky['rerun_pooled_weighted_f1']:.3f}   <- comparable to the honest arm")
    out(f"median candidate score     : {leaky['median_candidate_score']:.3f}")
    out(f"candidates evaluated       : {leaky['n_candidates']}")
    out()
    out("The first two lines are the pair that appears in a discontinued mlp_results.txt")
    out("as 'Best F1-Score' and 'Mean F1-Score'. They agree because they are the same")
    out("computation, which is why the second is not an independent estimate.")
    out()

    inflation = leaky["rerun_pooled_weighted_f1"] - honest["pooled_weighted_f1"]
    out("--- DELTA ---")
    out(f"leaky pooled - honest pooled (weighted F1)  : {inflation:+.3f}")
    out(f"leaky reported - honest pooled (weighted F1): {leaky['reported_best_score'] - honest['pooled_weighted_f1']:+.3f}")
    out(f"honest pooled accuracy - majority baseline  : {honest['pooled_accuracy'] - baseline:+.3f}")
    out()
    if honest["pooled_accuracy"] <= baseline:
        out("NOTE: the honest estimate does not beat the majority-class baseline.")

    (outdir / "protocol_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_groups": int(groups.nunique()),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
        "outer_cv": args.outer_cv,
        "grid": args.grid,
        "n_candidates": n_grid,
        "majority_baseline": baseline,
        "honest": {k: v for k, v in honest.items() if k not in {"y_true", "y_pred"}},
        "leaky": {k: v for k, v in leaky.items() if k not in {"y_true", "y_pred"}},
        "inflation_weighted_f1_pooled": inflation,
        "inflation_weighted_f1_reported": leaky["reported_best_score"] - honest["pooled_weighted_f1"],
    }
    (outdir / "protocol_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {outdir/'protocol_comparison.txt'} and .json")


if __name__ == "__main__":
    main()
