#!/usr/bin/env python3
# Canonical stage implementation: supported through presets plus run_pipeline.py.
# This file remains part of the pipeline contract, but it is not the app-level
# job submission entrypoint by itself.
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

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


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
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


class TorchMLPClassifierWrapper:
    def __init__(
        self,
        *,
        hidden_layer_sizes,
        activation,
        alpha,
        learning_rate_init,
        batch_size,
        device_name,
        max_epochs,
        patience,
        random_state,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch backend requested, but torch is not installed in this environment.")
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.batch_size = int(batch_size)
        self.device_name = device_name
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.classes_ = None
        self.class_to_index_ = None
        self.model_ = None
        self.device_ = None

    def _resolve_device(self):
        if self.device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        requested = torch.device(self.device_name)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for MLP, but torch.cuda.is_available() is false.")
        return requested

    def _activation_module(self):
        if self.activation == "relu":
            return nn.ReLU
        if self.activation == "tanh":
            return nn.Tanh
        raise ValueError(f"Unsupported activation for torch backend: {self.activation}")

    def _build_network(self, input_dim: int, output_dim: int):
        layers = []
        current_dim = input_dim
        activation_cls = self._activation_module()
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(activation_cls())
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_np = self.scaler.fit_transform(X.to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
        classes = sorted(pd.Series(y).astype(str).unique().tolist())
        class_to_index = {label: index for index, label in enumerate(classes)}
        y_idx = np.asarray([class_to_index[str(value)] for value in y], dtype=np.int64)

        self.classes_ = np.asarray(classes)
        self.class_to_index_ = class_to_index
        self.device_ = self._resolve_device()
        self.model_ = self._build_network(X_np.shape[1], len(classes)).to(self.device_)

        dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_idx))
        loader = DataLoader(
            dataset,
            batch_size=max(1, min(self.batch_size, len(dataset))),
            shuffle=True,
            num_workers=0,
            pin_memory=self.device_.type == "cuda",
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, self.patience // 4),
        )

        best_state = None
        best_loss = float("inf")
        epochs_without_improvement = 0

        for _epoch in range(self.max_epochs):
            self.model_.train()
            batch_losses = []
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device_, non_blocking=self.device_.type == "cuda")
                batch_y = batch_y.to(self.device_, non_blocking=self.device_.type == "cuda")

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")
            scheduler.step(epoch_loss)

            if epoch_loss + 1e-8 < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("TorchMLPClassifierWrapper must be fit before predict_proba.")
        if isinstance(X, pd.DataFrame):
            X_np = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X_np).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X_scaled).to(self.device_, non_blocking=self.device_.type == "cuda")
            logits = self.model_(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]


def _to_internal_params(best_params: dict) -> dict:
    if "hidden_layer_sizes" in best_params:
        return {
            "hidden_layer_sizes": tuple(best_params["hidden_layer_sizes"]),
            "activation": best_params["activation"],
            "alpha": float(best_params["alpha"]),
            "learning_rate_init": float(best_params["learning_rate_init"]),
            "batch_size": int(best_params["batch_size"]),
        }
    return {
        "hidden_layer_sizes": tuple(best_params["mlp__hidden_layer_sizes"]),
        "activation": best_params["mlp__activation"],
        "alpha": float(best_params["mlp__alpha"]),
        "learning_rate_init": float(best_params["mlp__learning_rate_init"]),
        "batch_size": int(best_params["mlp__batch_size"]),
    }


def _to_serializable_params(params: dict) -> dict:
    return {
        "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
        "activation": params["activation"],
        "alpha": float(params["alpha"]),
        "learning_rate_init": float(params["learning_rate_init"]),
        "batch_size": int(params["batch_size"]),
        "backend": mlp_backend,
        "device": mlp_device,
        "max_epochs": mlp_max_epochs,
        "patience": mlp_patience,
    }


def _resolve_param_grid(profile: str) -> dict:
    if profile == "expanded":
        return {
            "hidden_layer_sizes": [
                (8,),
                (16,),
                (32,),
                (64,),
                (128,),
                (16, 8),
                (32, 16),
                (64, 32),
                (128, 64),
                (32, 16, 8),
                (64, 32, 16),
                (128, 64, 32),
                (40, 20, 10, 5),
            ],
            "activation": ["relu", "tanh"],
            "alpha": [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
            "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "batch_size": [8, 16, 32],
        }
    return {
        "hidden_layer_sizes": [
            (8,),
            (16,),
            (32,),
            (64,),
            (16, 8),
            (32, 16),
            (64, 32),
            (32, 16, 8),
            (40, 20, 10, 5),
        ],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "batch_size": [8, 16, 32],
    }


def _maybe_balance_training_data(X: pd.DataFrame, y: pd.Series):
    if mlp_resampling == "none":
        return X, y
    if mlp_resampling != "oversample_minority":
        raise ValueError(f"Unsupported MLP resampling strategy: {mlp_resampling}")

    y_series = pd.Series(y).astype(str)
    class_counts = y_series.value_counts()
    if class_counts.empty or len(class_counts) < 2:
        return X, y

    max_count = int(class_counts.max())
    rng = np.random.default_rng(42)
    sampled_indices = []
    for label, count in class_counts.items():
        label_indices = y_series.index[y_series == label].to_numpy()
        sampled_indices.extend(label_indices.tolist())
        if count < max_count:
            extra = rng.choice(label_indices, size=max_count - int(count), replace=True)
            sampled_indices.extend(extra.tolist())

    if isinstance(X.index, pd.Index):
        balanced_X = X.loc[sampled_indices]
    else:
        balanced_X = X.iloc[sampled_indices]
    balanced_y = y.loc[sampled_indices]
    return balanced_X, balanced_y


def _selection_score(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> float:
    if mlp_selection_metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    if mlp_selection_metric == "macro_f1":
        return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    if mlp_selection_metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported MLP selection metric: {mlp_selection_metric}")


def _build_model(params, *, backend, device_name, max_epochs, patience, random_state):
    if backend == "torch":
        return TorchMLPClassifierWrapper(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            device_name=device_name,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
        )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=params["hidden_layer_sizes"],
                    activation=params["activation"],
                    alpha=params["alpha"],
                    learning_rate_init=params["learning_rate_init"],
                    batch_size=params["batch_size"],
                    solver="adam",
                    learning_rate="adaptive",
                    max_iter=max_epochs,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _fit_inner_model(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series):
    logo = LeaveOneGroupOut()
    all_labels = sorted(y_train.unique())
    group_values = groups_train.to_numpy()

    param_grid = _resolve_param_grid(mlp_grid_profile)

    best_score = -np.inf
    best_params = None

    for hidden_sizes, activation, alpha, learning_rate_init, batch_size in product(
        param_grid["hidden_layer_sizes"],
        param_grid["activation"],
        param_grid["alpha"],
        param_grid["learning_rate_init"],
        param_grid["batch_size"],
    ):
        params = {
            "hidden_layer_sizes": hidden_sizes,
            "activation": activation,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "batch_size": batch_size,
        }
        fold_scores = []
        for inner_train_idx, inner_test_idx in logo.split(X_train, y_train, group_values):
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_test = X_train.iloc[inner_test_idx]
            y_inner_train = y_train.iloc[inner_train_idx]
            y_inner_test = y_train.iloc[inner_test_idx]

            X_inner_train_fit, y_inner_train_fit = _maybe_balance_training_data(X_inner_train, y_inner_train)
            model = _build_model(
                params,
                backend=mlp_backend,
                device_name=mlp_device,
                max_epochs=mlp_max_epochs,
                patience=mlp_patience,
                random_state=42,
            )
            model.fit(X_inner_train_fit, y_inner_train_fit)
            y_pred = model.predict(X_inner_test)
            fold_scores.append(
                _selection_score(y_inner_test, y_pred, all_labels)
            )

        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("Inner model selection failed.")
    best_model = _build_model(
        best_params,
        backend=mlp_backend,
        device_name=mlp_device,
        max_epochs=mlp_max_epochs,
        patience=mlp_patience,
        random_state=42,
    )
    return best_model, best_params


def _load_fixed_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "best_params" in payload:
        payload = payload["best_params"]
    return _to_internal_params(payload)


def _save_fixed_params(path: Path, params: dict, *, selection_score: float | None = None, selection_scope: str = "grouped_full_data"):
    payload = {
        "selection_scope": selection_scope,
        "selection_metric": mlp_selection_metric,
        "grid_profile": mlp_grid_profile,
        "resampling": mlp_resampling,
        "best_params": _to_serializable_params(params),
    }
    if selection_score is not None:
        payload["selection_score"] = float(selection_score)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _search_best_params(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series):
    model, params = _fit_inner_model(X_train, y_train, groups_train)
    logo = LeaveOneGroupOut()
    all_labels = sorted(y_train.unique())
    fold_scores = []
    for inner_train_idx, inner_test_idx in logo.split(X_train, y_train, groups_train.to_numpy()):
        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train.iloc[inner_train_idx]
        X_inner_train_fit, y_inner_train_fit = _maybe_balance_training_data(X_inner_train, y_inner_train)
        search_model = _build_model(
            params,
            backend=mlp_backend,
            device_name=mlp_device,
            max_epochs=mlp_max_epochs,
            patience=mlp_patience,
            random_state=42,
        )
        search_model.fit(X_inner_train_fit, y_inner_train_fit)
        y_pred = search_model.predict(X_train.iloc[inner_test_idx])
        fold_scores.append(
            _selection_score(y_train.iloc[inner_test_idx], y_pred, all_labels)
        )
    return model, params, float(np.mean(fold_scores)) if fold_scores else float("nan")


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
skip_shap = os.environ.get("NICHERUNNER_SKIP_SHAP", "0") == "1"
mlp_backend = os.environ.get("NICHERUNNER_MLP_BACKEND", "sklearn").strip().lower()
mlp_device = os.environ.get("NICHERUNNER_MLP_DEVICE", "auto").strip().lower()
mlp_max_epochs = int(os.environ.get("NICHERUNNER_MLP_MAX_EPOCHS", "1000"))
mlp_patience = int(os.environ.get("NICHERUNNER_MLP_PATIENCE", "20"))
mlp_selection_metric = os.environ.get("NICHERUNNER_MLP_SELECTION_METRIC", "weighted_f1").strip().lower()
mlp_grid_profile = os.environ.get("NICHERUNNER_MLP_GRID_PROFILE", "default").strip().lower()
mlp_resampling = os.environ.get("NICHERUNNER_MLP_RESAMPLING", "none").strip().lower()
mlp_mode = os.environ.get("NICHERUNNER_MLP_MODE", "nested_cv").strip().lower()
mlp_fixed_params_path_env = os.environ.get("NICHERUNNER_MLP_FIXED_PARAMS_PATH", "").strip()
mlp_best_params_out_env = os.environ.get("NICHERUNNER_MLP_BEST_PARAMS_OUT", "").strip()

if mlp_backend not in {"sklearn", "torch"}:
    raise ValueError("NICHERUNNER_MLP_BACKEND must be either 'sklearn' or 'torch'.")
if mlp_mode not in {"nested_cv", "tune_once", "evaluate_fixed", "explain"}:
    raise ValueError("NICHERUNNER_MLP_MODE must be one of: nested_cv, tune_once, evaluate_fixed, explain.")
if mlp_selection_metric not in {"weighted_f1", "macro_f1", "balanced_accuracy"}:
    raise ValueError("NICHERUNNER_MLP_SELECTION_METRIC must be one of: weighted_f1, macro_f1, balanced_accuracy.")
if mlp_grid_profile not in {"default", "expanded"}:
    raise ValueError("NICHERUNNER_MLP_GRID_PROFILE must be either 'default' or 'expanded'.")
if mlp_resampling not in {"none", "oversample_minority"}:
    raise ValueError("NICHERUNNER_MLP_RESAMPLING must be either 'none' or 'oversample_minority'.")

mlp_fixed_params_path = Path(mlp_fixed_params_path_env) if mlp_fixed_params_path_env else (output_dir / "fixed_params.json")
mlp_best_params_out = Path(mlp_best_params_out_env) if mlp_best_params_out_env else (output_dir / "fixed_params.json")

original_stdout = sys.stdout
sys.stdout = open(output_path, "w", encoding="utf-8")

try:
    print("--- Starting Leakage-Safe Nested Grouped Evaluation ---")
    print(f"Evaluation unit: {mlp_unit}")
    print("Outer CV mode: logo_grouped")
    print("Inner CV mode: nested training-only tuning with grouped splits")
    print(f"Skip SHAP: {skip_shap}")
    print(f"MLP backend: {mlp_backend}")
    print(f"MLP device: {mlp_device}")
    print(f"MLP max epochs: {mlp_max_epochs}")
    print(f"MLP selection metric: {mlp_selection_metric}")
    print(f"MLP grid profile: {mlp_grid_profile}")
    print(f"MLP resampling: {mlp_resampling}")
    print(f"MLP mode: {mlp_mode}")
    print(f"Fixed params path: {mlp_fixed_params_path}")
    print(f"Best params out: {mlp_best_params_out}")
    if mlp_backend == "torch":
        print(f"Torch available: {TORCH_AVAILABLE}")
        if TORCH_AVAILABLE:
            print(f"CUDA available: {torch.cuda.is_available()}")

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
    final_params_for_explain = None

    if mlp_mode == "tune_once":
        print("--- Running one grouped hyperparameter search on full data ---")
        selected_features_full = _select_features(
            nmf_props,
            enrichment_frame,
            niche_gene_frame,
            y,
            top_enrichment=top_enrichment,
            top_niche_gene=top_niche_gene,
        )
        X_explain = X_full[selected_features_full]
        _model, best_params, search_score = _search_best_params(X_explain, y, groups)
        _save_fixed_params(mlp_best_params_out, best_params, selection_score=search_score)
        print(f"Saved fixed params to: {mlp_best_params_out}")
        print(f"Grouped full-data selection score for tuned params: {search_score:.3f}")
        print(f"Best params: {_to_serializable_params(best_params)}")
        final_params_for_explain = best_params

    elif mlp_mode in {"nested_cv", "evaluate_fixed"}:
        fixed_params = None
        if mlp_mode == "evaluate_fixed":
            if not mlp_fixed_params_path.exists():
                raise FileNotFoundError(f"Fixed params file not found: {mlp_fixed_params_path}")
            fixed_params = _load_fixed_params(mlp_fixed_params_path)
            print(f"Loaded fixed params: {_to_serializable_params(fixed_params)}")

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_full, y, groups), start=1):
            train_items = X_full.index[train_idx]
            test_items = X_full.index[test_idx]
            train_groups = groups.iloc[train_idx]
            test_groups = groups.iloc[test_idx]
            print(f"--- Processing Outer Fold {fold_idx}/{len(unique_groups)} ---")
            print(f"Train groups: {sorted(train_groups.astype(str).unique().tolist())}")
            print(f"Test groups: {sorted(test_groups.astype(str).unique().tolist())}")
            print(f"Train rows: {len(train_items)} | Test rows: {len(test_items)}")
            sys.stdout.flush()

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

            if mlp_mode == "nested_cv":
                model, best_params = _fit_inner_model(X_train, y_train, train_groups)
            else:
                best_params = dict(fixed_params)
                model = _build_model(
                    best_params,
                    backend=mlp_backend,
                    device_name=mlp_device,
                    max_epochs=mlp_max_epochs,
                    patience=mlp_patience,
                    random_state=42,
                )
            X_train_fit, y_train_fit = _maybe_balance_training_data(X_train, y_train)
            model.fit(X_train_fit, y_train_fit)
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
                    "best_params": _to_serializable_params(best_params),
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

        if mlp_mode == "nested_cv":
            selected_features_full = _select_features(
                nmf_props,
                enrichment_frame,
                niche_gene_frame,
                y,
                top_enrichment=top_enrichment,
                top_niche_gene=top_niche_gene,
            )
            X_explain = X_full[selected_features_full]
            _final_model, final_params_for_explain, _score = _search_best_params(X_explain, y, groups)
            _save_fixed_params(mlp_best_params_out, final_params_for_explain, selection_score=_score)
        else:
            final_params_for_explain = fixed_params

    if not skip_shap and mlp_mode in {"nested_cv", "evaluate_fixed", "explain"}:
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
        if mlp_mode == "explain":
            if not mlp_fixed_params_path.exists():
                raise FileNotFoundError(f"Fixed params file not found: {mlp_fixed_params_path}")
            final_params_for_explain = _load_fixed_params(mlp_fixed_params_path)
        if final_params_for_explain is None:
            _final_model, final_params_for_explain, _score = _search_best_params(X_explain, y, groups)
            _save_fixed_params(mlp_best_params_out, final_params_for_explain, selection_score=_score)
        final_model = _build_model(
            final_params_for_explain,
            backend=mlp_backend,
            device_name=mlp_device,
            max_epochs=mlp_max_epochs,
            patience=mlp_patience,
            random_state=42,
        )
        X_explain_fit, y_explain_fit = _maybe_balance_training_data(X_explain, y)
        final_model.fit(X_explain_fit, y_explain_fit)
        print(f"Final explanatory params: {_to_serializable_params(final_params_for_explain)}")

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
    elif skip_shap:
        print("\n--- SHAP skipped by configuration ---")

finally:
    sys.stdout.close()
    sys.stdout = original_stdout
