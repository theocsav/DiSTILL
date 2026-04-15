# -*- coding: utf-8 -*-
import cell2location as c2l
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import scanpy as sc
import pandas as pd
import numpy as np
import scvi
import anndata as ad
import os
import json
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scikit_posthocs import posthoc_dunn
from kneed import KneeLocator
from collections import defaultdict
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from sklearn.neighbors import KDTree
from scipy.stats import kruskal
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import statsmodels.api as sm
import matplotlib.ticker as ticker
from scipy.sparse import issparse
from sklearn.feature_selection import mutual_info_classif
import shap

from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

# --- Define the output directory ---
feature_input_dir = os.environ.get("NICHERUNNER_OUTPUT_DIR", "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis")
output_dir = os.environ.get("NICHERUNNER_MLP_OUTPUT_DIR", os.path.join(feature_input_dir, "MLP_44Features"))
cv_mode = os.environ.get("NICHERUNNER_CV_MODE", "sgkf3").strip().lower()
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'mlp_results.txt')

# --- Redirect print output to the file ---
import sys
original_stdout = sys.stdout
sys.stdout = open(output_path, 'w')

try:
    # --- Load the pre-calculated features and labels ---
    X = pd.read_parquet(os.path.join(feature_input_dir, 'reduced_features_final_15.parquet'))
    y = pd.read_parquet(os.path.join(feature_input_dir, 'targets_y.parquet')).squeeze()
    groups = pd.read_parquet(os.path.join(feature_input_dir, 'groups.parquet')).squeeze()

    if cv_mode == "loocv":
        cv_splitter = LeaveOneGroupOut()
        cv_label = "Leave-One-Group-Out Cross-Validation"
        split_iterator = lambda: cv_splitter.split(X, y, groups)
        n_cv_splits = groups.nunique()
    else:
        cv_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        cv_label = "3-Fold Stratified Group Cross-Validation"
        split_iterator = lambda: cv_splitter.split(X, y, groups)
        n_cv_splits = 3

    # --- 4. Hyperparameter Tuning and Cross-Validation ---
    print("--- Starting Hyperparameter Search with RandomizedSearchCV ---")
    print(f"CV mode: {cv_mode} ({cv_label})")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            solver='adam', learning_rate='adaptive', max_iter=1000, random_state=42
        ))
    ])
    param_distributions = {
        'mlp__hidden_layer_sizes': [
    		(32, 16, 8),            # A small, tapering network
    		(44, 22),               # Starts at the number of features, then tapers
    		(50, 25, 12),           # A slightly larger tapering option
    		(64, 32),               # A wide and shallow network
    		(40, 20, 10, 5),        # A deeper network with a small number of neurons
   		    (25,),                  # A single, small hidden layer
    		(50, 50),               # A network with consistent width
	    ],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': loguniform(1e-5, 1e-1),
        'mlp__batch_size': [2, 4, 8, 16]
    }
    random_search = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_distributions, n_iter=30000, cv=cv_splitter,
        scoring='f1_weighted', n_jobs=-1, random_state=42, verbose=1
    )
    random_search.fit(X, y, groups=groups)

    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best F1-Score: {random_search.best_score_:.4f}")
    print("Best Hyperparameters Found:")
    print(random_search.best_params_)
    # Save best parameters to a JSON file
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(random_search.best_params_, f, indent=4)

    # --- 5. Run a final cross-validation with the best model ---
    print(f"\n--- Running {cv_label} with Best Model ---")
    best_pipeline = random_search.best_estimator_
    all_y_true = []
    all_y_pred = []
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    all_labels = sorted(y.unique())

    for i, (train_idx, test_idx) in enumerate(split_iterator()):
        print(f"--- Processing Fold {i+1}/{n_cv_splits} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)
        
        fold_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        fold_precisions.append(precision_score(y_test, y_pred, labels=all_labels, average='weighted', zero_division=0))
        fold_recalls.append(recall_score(y_test, y_pred, labels=all_labels, average='weighted', zero_division=0))
        fold_f1_scores.append(f1_score(y_test, y_pred, labels=all_labels, average='weighted', zero_division=0))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # --- 6. Print Final Report and Confusion Matrix ---
    print("\n--- Final Performance Report ---")
    print(f"Accuracies for each fold: {np.round(fold_accuracies, 3)}")
    print(f"Precisions for each fold: {np.round(fold_precisions, 3)}")
    print(f"Recalls for each fold: {np.round(fold_recalls, 3)}")
    print(f"F1-Scores for each fold: {np.round(fold_f1_scores, 3)}")
    print("\n--- Mean and Standard Deviation ---")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.3f} (± {np.std(fold_accuracies):.3f})")
    print(f"Mean Precision: {np.mean(fold_precisions):.3f} (± {np.std(fold_precisions):.3f})")
    print(f"Mean Recall: {np.mean(fold_recalls):.3f} (± {np.std(fold_recalls):.3f})")
    print(f"Mean F1-Score: {np.mean(fold_f1_scores):.3f} (± {np.std(fold_f1_scores):.3f})")
    print("\n--- Overall Classification Report ---")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))
    print("\n--- Overall Confusion Matrix ---")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=np.unique(all_y_true))
    cm_df = pd.DataFrame(cm, index=np.unique(all_y_true), columns=np.unique(all_y_true))
    print(cm_df)
    
    # Save confusion matrix to a CSV file
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))

    # Legacy permutation-importance block retained for reference.
    # print("\n--- Permutation Importance ---")
    # best_pipeline.fit(X, y)
    # pi = permutation_importance(
    #     best_pipeline,
    #     X,
    #     y,
    #     scoring='f1_weighted',
    #     n_repeats=200,
    #     random_state=42,
    #     n_jobs=-1,
    # )
    # pi_df = pd.DataFrame({
    #     'feature': X.columns,
    #     'importance_mean': pi.importances_mean,
    #     'importance_std': pi.importances_std,
    # }).sort_values('importance_mean', ascending=False)
    # pi_df.to_csv(os.path.join(output_dir, 'permutation_importance.csv'), index=False)
    # print(pi_df.head(20).to_string(index=False))
    #
    # top_n = min(20, len(pi_df))
    # pi_plot = pi_df.head(top_n).iloc[::-1]
    # plt.figure(figsize=(10, max(6, top_n * 0.35)))
    # colors = ['#1f77b4' if value >= 0 else '#d62728' for value in pi_plot['importance_mean']]
    # plt.barh(
    #     pi_plot['feature'],
    #     pi_plot['importance_mean'],
    #     xerr=pi_plot['importance_std'],
    #     color=colors,
    #     edgecolor='black',
    #     alpha=0.85,
    # )
    # plt.xlabel('Mean decrease in weighted F1 after permutation')
    # plt.ylabel('Feature')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'permutation_importance_top20.png'), dpi=200, bbox_inches='tight')
    # plt.close()

    print("\n--- SHAP Analysis ---")
    best_pipeline.fit(X, y)
    class_labels = list(best_pipeline.classes_)
    positive_class = 'systemic_sclerosis' if 'systemic_sclerosis' in class_labels else class_labels[-1]
    positive_idx = class_labels.index(positive_class)

    def predict_proba_fn(data):
        frame = pd.DataFrame(data, columns=X.columns)
        return best_pipeline.predict_proba(frame)

    explainer = shap.KernelExplainer(
        predict_proba_fn,
        X,
        feature_names=list(X.columns),
    )
    shap_values = explainer.shap_values(X, nsamples=min(200, max(50, X.shape[1] * 10)))

    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[positive_idx])
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_matrix = shap_array[:, :, positive_idx]
        else:
            shap_matrix = shap_array

    shap_df = pd.DataFrame(shap_matrix, columns=X.columns, index=X.index)
    shap_df.to_csv(os.path.join(output_dir, 'shap_values_positive_class.csv'), index=True)

    shap_importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_matrix).mean(axis=0),
        'mean_shap': shap_matrix.mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance_df.to_csv(os.path.join(output_dir, 'shap_importance.csv'), index=False)
    print(f"Positive class used for SHAP: {positive_class}")
    print(shap_importance_df.head(20).to_string(index=False))

    top_n = min(20, len(shap_importance_df))
    shap_plot = shap_importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    colors = ['#1f77b4' if value >= 0 else '#d62728' for value in shap_plot['mean_shap']]
    plt.barh(
        shap_plot['feature'],
        shap_plot['mean_abs_shap'],
        color=colors,
        edgecolor='black',
        alpha=0.85,
    )
    plt.xlabel('Mean absolute SHAP value')
    plt.ylabel('Feature')
    plt.title(f'SHAP feature importance for {positive_class}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_importance_top20.png'), dpi=200, bbox_inches='tight')
    plt.close()

finally:
    # --- Restore original stdout ---
    sys.stdout.close()
    sys.stdout = original_stdout
