#!/usr/bin/env python3
import json
import os
import sys
from itertools import product
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _pick_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"None of the expected columns were found: {candidates}")


def _read_table(path_stem: Path) -> pd.DataFrame:
    parquet_path = path_stem.with_suffix(".parquet")
    csv_path = path_stem.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists.")


def _load_metadata_from_h5ad(h5ad_path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs.copy()
    patient_col = _pick_column(list(obs.columns), ["patient"])
    disease_col = _pick_column(list(obs.columns), ["disease_state", "Disease_State", "Disease/Health State"])
    if "unique_fov" in obs.columns:
        fov_key = obs["unique_fov"].astype(str)
    elif "fov" in obs.columns:
        fov_key = obs[patient_col].astype(str) + "_" + obs["fov"].astype(str)
    else:
        raise RuntimeError("cosmx_with_nmf.h5ad is missing 'fov' or 'unique_fov' needed for FOV aggregation.")
    return pd.DataFrame(
        {
            "patient": obs[patient_col].astype(str).values,
            "Disease_State": obs[disease_col].astype(str).values,
            "fov_key": fov_key.values,
        },
        index=obs.index.astype(str),
    )


def _safe_mutual_info(X: pd.DataFrame, y_codes: np.ndarray) -> pd.Series:
    usable = X.loc[:, X.var(axis=0) > 0]
    if usable.empty:
        return pd.Series(dtype=float)
    mi = mutual_info_classif(usable.to_numpy(), y_codes, random_state=42)
    return pd.Series(mi, index=usable.columns).sort_values(ascending=False)


def _select_features(
    nmf_props: pd.DataFrame,
    enrichment_frame: pd.DataFrame,
    niche_gene_frame: pd.DataFrame,
    y_train: pd.Series,
    top_enrichment: int,
    top_niche_gene: int,
):
    y_codes = pd.Categorical(y_train).codes

    selected = list(nmf_props.columns)

    enrichment_scores = _safe_mutual_info(enrichment_frame.loc[y_train.index], y_codes)
    selected.extend(list(enrichment_scores.head(max(0, top_enrichment)).index))

    niche_gene_scores = _safe_mutual_info(niche_gene_frame.loc[y_train.index], y_codes)
    selected.extend(list(niche_gene_scores.head(max(0, top_niche_gene)).index))

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(selected))


def _score_fold(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def _fit_inner_model(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series):
    logo = LeaveOneGroupOut()
    all_labels = sorted(y_train.unique())
    group_values = groups_train.to_numpy()

    param_grid = {
        "hidden_layer_sizes": [(8,), (16,), (16, 8), (24, 12)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "batch_size": [2, 4],
    }

    best_score = -np.inf
    best_params = None
    best_model = None

    for hidden_sizes, activation, alpha, batch_size in product(
        param_grid["hidden_layer_sizes"],
        param_grid["activation"],
        param_grid["alpha"],
        param_grid["batch_size"],
    ):
        fold_scores = []
        for inner_train_idx, inner_test_idx in logo.split(X_train, y_train, group_values):
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_test = X_train.iloc[inner_test_idx]
            y_inner_train = y_train.iloc[inner_train_idx]
            y_inner_test = y_train.iloc[inner_test_idx]

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=hidden_sizes,
                            activation=activation,
                            alpha=alpha,
                            batch_size=batch_size,
                            solver="adam",
                            learning_rate="adaptive",
                            max_iter=1000,
                            random_state=42,
                        ),
                    ),
                ]
            )
            model.fit(X_inner_train, y_inner_train)
            y_pred = model.predict(X_inner_test)
            fold_scores.append(
                f1_score(y_inner_test, y_pred, labels=all_labels, average="weighted", zero_division=0)
            )

        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                "mlp__hidden_layer_sizes": hidden_sizes,
                "mlp__activation": activation,
                "mlp__alpha": alpha,
                "mlp__batch_size": batch_size,
            }
            best_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=hidden_sizes,
                            activation=activation,
                            alpha=alpha,
                            batch_size=batch_size,
                            solver="adam",
                            learning_rate="adaptive",
                            max_iter=1000,
                            random_state=42,
                        ),
                    ),
                ]
            )

    if best_model is None or best_params is None:
        raise RuntimeError("Inner model selection failed.")
    return best_model, best_params


def _prepare_patient_frames(feature_input_dir: Path, source_output_dir: Path):
    combined = pd.read_parquet(feature_input_dir / "combined_features_filtered.parquet")
    nmf_props = combined.loc[:, [c for c in combined.columns if str(c).startswith("nmf_prop_")]].copy()
    y = pd.read_parquet(feature_input_dir / "targets_y.parquet").squeeze().astype(str)
    groups = pd.read_parquet(feature_input_dir / "groups.parquet").squeeze().astype(str)

    metadata = _load_metadata_from_h5ad(source_output_dir / "cosmx_with_nmf.h5ad")
    fov_meta = metadata.groupby("fov_key").agg({"patient": "first", "Disease_State": "first"})

    enrichment_fov = _read_table(source_output_dir / "enrichment_features_fov")
    enrichment_fov.index = enrichment_fov.index.astype(str)
    enrichment_patient = enrichment_fov.join(fov_meta[["patient"]], how="left").groupby("patient").mean()
    enrichment_patient = enrichment_patient.reindex(y.index).fillna(0.0)

    niche_gene_fov = _read_table(source_output_dir / "niche_gene_features_fov")
    niche_gene_fov.index = niche_gene_fov.index.astype(str)
    niche_gene_patient = niche_gene_fov.join(fov_meta[["patient"]], how="left").groupby("patient").mean()
    niche_gene_patient = niche_gene_patient.reindex(y.index).fillna(0.0)

    return nmf_props, enrichment_patient, niche_gene_patient, y, groups


def _prepare_fov_frames(feature_input_dir: Path, source_output_dir: Path):
    combined = pd.read_parquet(feature_input_dir / "combined_features_filtered.parquet")
    combined.index = combined.index.astype(str)
    nmf_props = combined.loc[:, [c for c in combined.columns if str(c).startswith("nmf_prop_")]].copy()
    y = pd.read_parquet(feature_input_dir / "targets_y.parquet").squeeze().astype(str)
    groups = pd.read_parquet(feature_input_dir / "groups.parquet").squeeze().astype(str)
    y.index = y.index.astype(str)
    groups.index = groups.index.astype(str)

    enrichment_fov = _read_table(source_output_dir / "enrichment_features_fov")
    enrichment_fov.index = enrichment_fov.index.astype(str)
    enrichment_fov = enrichment_fov.reindex(combined.index).fillna(0.0)

    niche_gene_fov = _read_table(source_output_dir / "niche_gene_features_fov")
    niche_gene_fov.index = niche_gene_fov.index.astype(str)
    niche_gene_fov = niche_gene_fov.reindex(combined.index).fillna(0.0)

    return nmf_props, enrichment_fov, niche_gene_fov, y.reindex(combined.index), groups.reindex(combined.index)


feature_input_dir = Path(
    os.environ.get(
        "NICHERUNNER_OUTPUT_DIR",
        "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis",
    )
)
source_output_dir = Path(os.environ.get("NICHERUNNER_SOURCE_OUTPUT_DIR", str(feature_input_dir)))
mlp_unit = os.environ.get("NICHERUNNER_MLP_UNIT", "patient").strip().lower()
output_dir = Path(os.environ.get("NICHERUNNER_MLP_OUTPUT_DIR", str(feature_input_dir / "MLP_LeakageSafe")))
os.makedirs(output_dir, exist_ok=True)
output_path = output_dir / "mlp_results.txt"

top_enrichment = int(os.environ.get("NICHERUNNER_TOP_ENRICHMENT_FEATURES", "5"))
top_niche_gene = int(os.environ.get("NICHERUNNER_TOP_NICHE_GENE_FEATURES", "20"))

original_stdout = sys.stdout
sys.stdout = open(output_path, "w", encoding="utf-8")

try:
    print("--- Starting Leakage-Safe Nested Grouped Evaluation ---")
    print(f"Evaluation unit: {mlp_unit}")
    print("Outer CV mode: logo_grouped")
    print("Inner CV mode: nested training-only tuning with grouped splits")

    if mlp_unit == "fov":
        nmf_props, enrichment_frame, niche_gene_frame, y, groups = _prepare_fov_frames(
            feature_input_dir, source_output_dir
        )
    else:
        nmf_props, enrichment_frame, niche_gene_frame, y, groups = _prepare_patient_frames(
            feature_input_dir, source_output_dir
        )

    X_full = nmf_props.join(enrichment_frame, how="left").join(niche_gene_frame, how="left").fillna(0.0)

    logo = LeaveOneGroupOut()
    all_y_true = []
    all_y_pred = []
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    fold_records = []
    all_labels = sorted(y.unique())

    unique_groups = list(pd.Index(groups.astype(str).unique()))
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_full, y, groups), start=1):
        train_items = X_full.index[train_idx]
        test_items = X_full.index[test_idx]
        train_groups = groups.iloc[train_idx]
        test_groups = groups.iloc[test_idx]
        print(f"--- Processing Outer Fold {fold_idx}/{len(unique_groups)} ---")
        print(f"Train groups: {sorted(train_groups.astype(str).unique().tolist())}")
        print(f"Test groups: {sorted(test_groups.astype(str).unique().tolist())}")
        print(f"Train rows: {len(train_items)} | Test rows: {len(test_items)}")

        y_train = y.loc[train_items]
        y_test = y.loc[test_items]

        selected_features = _select_features(
            nmf_props.loc[train_items],
            enrichment_frame.loc[train_items],
            niche_gene_frame.loc[train_items],
            y_train,
            top_enrichment=top_enrichment,
            top_niche_gene=top_niche_gene,
        )

        X_train = X_full.loc[train_items, selected_features]
        X_test = X_full.loc[test_items, selected_features]

        model, best_params = _fit_inner_model(X_train, y_train, train_groups)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores = _score_fold(y_test, y_pred, all_labels)
        fold_accuracies.append(scores["accuracy"])
        fold_precisions.append(scores["precision"])
        fold_recalls.append(scores["recall"])
        fold_f1_scores.append(scores["f1"])
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fold_records.append(
            {
                "outer_fold": fold_idx,
                "train_groups": sorted(train_groups.astype(str).unique().tolist()),
                "test_groups": sorted(test_groups.astype(str).unique().tolist()),
                "train_rows": int(len(train_items)),
                "test_rows": int(len(test_items)),
                "selected_feature_count": len(selected_features),
                "selected_features": selected_features,
                "best_params": best_params,
            }
        )

    print("\n--- Final Performance Report ---")
    print(f"Accuracies for each fold: {np.round(fold_accuracies, 3)}")
    print(f"Precisions for each fold: {np.round(fold_precisions, 3)}")
    print(f"Recalls for each fold: {np.round(fold_recalls, 3)}")
    print(f"F1-Scores for each fold: {np.round(fold_f1_scores, 3)}")
    print("\n--- Mean and Standard Deviation ---")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.3f} (+/- {np.std(fold_accuracies):.3f})")
    print(f"Mean Precision: {np.mean(fold_precisions):.3f} (+/- {np.std(fold_precisions):.3f})")
    print(f"Mean Recall: {np.mean(fold_recalls):.3f} (+/- {np.std(fold_recalls):.3f})")
    print(f"Mean F1-Score: {np.mean(fold_f1_scores):.3f} (+/- {np.std(fold_f1_scores):.3f})")
    print("\n--- Overall Classification Report ---")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))
    print("\n--- Overall Confusion Matrix ---")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(all_y_true))
    cm_df = pd.DataFrame(cm, index=np.unique(all_y_true), columns=np.unique(all_y_true))
    print(cm_df)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as handle:
        json.dump({"outer_folds": fold_records}, handle, indent=2)
    pd.DataFrame(
        [
            {
                "outer_fold": record["outer_fold"],
                "train_groups": ",".join(record["train_groups"]),
                "test_groups": ",".join(record["test_groups"]),
                "train_rows": record["train_rows"],
                "test_rows": record["test_rows"],
                "selected_feature_count": record["selected_feature_count"],
                "selected_features": "|".join(record["selected_features"]),
            }
            for record in fold_records
        ]
    ).to_csv(output_dir / "selected_features_by_fold.csv", index=False)

    print("\n--- Explanatory Full-Data Model (not part of unbiased evaluation) ---")
    selected_features_full = _select_features(
        nmf_props,
        enrichment_frame,
        niche_gene_frame,
        y,
        top_enrichment=top_enrichment,
        top_niche_gene=top_niche_gene,
    )
    X_explain = X_full[selected_features_full]
    final_model, final_params = _fit_inner_model(X_explain, y, groups)
    final_model.fit(X_explain, y)
    print(f"Final explanatory params: {final_params}")

    class_labels = list(final_model.classes_)
    positive_class = "systemic_sclerosis" if "systemic_sclerosis" in class_labels else class_labels[-1]
    positive_idx = class_labels.index(positive_class)

    def predict_proba_fn(data):
        frame = pd.DataFrame(data, columns=X_explain.columns)
        return final_model.predict_proba(frame)

    explainer = shap.KernelExplainer(predict_proba_fn, X_explain, feature_names=list(X_explain.columns))
    shap_values = explainer.shap_values(X_explain, nsamples=min(200, max(50, X_explain.shape[1] * 10)))

    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[positive_idx])
    else:
        shap_array = np.asarray(shap_values)
        shap_matrix = shap_array[:, :, positive_idx] if shap_array.ndim == 3 else shap_array

    shap_df = pd.DataFrame(shap_matrix, columns=X_explain.columns, index=X_explain.index)
    shap_df.to_csv(output_dir / "shap_values_positive_class.csv", index=True)

    shap_importance_df = pd.DataFrame(
        {
            "feature": X_explain.columns,
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
            "mean_shap": shap_matrix.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance_df.to_csv(output_dir / "shap_importance.csv", index=False)
    print(f"Positive class used for SHAP: {positive_class}")
    print(shap_importance_df.head(20).to_string(index=False))

    top_n = min(20, len(shap_importance_df))
    shap_plot = shap_importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in shap_plot["mean_shap"]]
    plt.barh(shap_plot["feature"], shap_plot["mean_abs_shap"], color=colors, edgecolor="black", alpha=0.85)
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"SHAP feature importance for {positive_class}")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_top20.png", dpi=200, bbox_inches="tight")
    plt.close()

finally:
    sys.stdout.close()
    sys.stdout = original_stdout
