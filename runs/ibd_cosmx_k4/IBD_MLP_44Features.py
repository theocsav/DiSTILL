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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy import stats
import statsmodels.api as sm
import matplotlib.ticker as ticker
from scipy.sparse import issparse
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform

# --- Define the output directory ---
output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis/MLP_44Features"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'mlp_results.txt')

# --- Redirect print output to the file ---
import sys
original_stdout = sys.stdout
sys.stdout = open(output_path, 'w')

try:
    # --- Load the pre-calculated features and labels ---
    feature_input_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis"
    X = pd.read_parquet(os.path.join(feature_input_dir, 'reduced_features_final_15.parquet'))
    y = pd.read_parquet(os.path.join(feature_input_dir, 'targets_y.parquet')).squeeze()
    groups = pd.read_parquet(os.path.join(feature_input_dir, 'groups.parquet')).squeeze()

    # --- 4. Hyperparameter Tuning and Cross-Validation ---
    print("--- Starting Hyperparameter Search with RandomizedSearchCV ---")
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
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_distributions, n_iter=30000, cv=sgkf,
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
    print("\n--- Running 3-Fold Stratified Group Cross-Validation with Best Model ---")
    best_pipeline = random_search.best_estimator_
    all_y_true = []
    all_y_pred = []
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    all_labels = sorted(y.unique())

    for i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"--- Processing Fold {i+1}/3 ---")
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

finally:
    # --- Restore original stdout ---
    sys.stdout.close()
    sys.stdout = original_stdout