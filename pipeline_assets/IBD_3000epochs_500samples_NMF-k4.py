# -*- coding: utf-8 -*-
import cell2location as c2l
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import scanpy as sc
import pandas as pd
import numpy as np
import math
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
import scipy.sparse
import re

def choose_reference_obs_key(obs: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for key in candidates:
        if key in obs.columns:
            return key
    raise KeyError(
        f"Reference h5ad is missing a usable {purpose} column. Tried: {', '.join(candidates)}"
    )

# --- Define Paths --------------------------------------------------------------------------------

reference_h5ad_path = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/scRNA/combined_10x_reference_final.h5ad"
cosmx_h5ad_path = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/GSE234713_CosMx_combined.h5ad"
ref_model_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/cell2location_models"
os.makedirs(ref_model_dir, exist_ok=True) # Ensure the directory exists
ref_model_path = os.path.join(ref_model_dir, "cell2location_reference_model_3000ep_500samp_NMF-k4")


####################################################################################################
# --- Step 1: Load and Initial Gene Alignment for BOTH adata_st and adata_ref ---------------------
####################################################################################################

print("--- Step 1: Loading and Initial Gene Alignment for both adata_st and adata_ref ---")

# Load adata_st (spatial data)
adata_st = sc.read(cosmx_h5ad_path)
adata_st.var_names_make_unique()
print(f"Loaded adata_st shape: {adata_st.shape}")

# Ensure MT genes are removed from adata_st
if "MT_gene" not in adata_st.var.columns or not np.any(adata_st.var["MT_gene"]):
    adata_st.var["MT_gene"] = [gene.startswith("MT-") for gene in adata_st.var_names]
    adata_st = adata_st[:, ~adata_st.var["MT_gene"].values].copy()
    print("MT genes ensured to be removed from adata_st.")
else:
    print("adata_st already processed for MT genes.")

# Store raw counts for adata_st. This is crucial for later steps.
adata_st.raw = adata_st.copy()
print("adata_st.raw created for raw counts.")


# Load adata_ref (single-cell reference)
adata_ref = sc.read(reference_h5ad_path)
adata_ref.var_names_make_unique()
print(f"Loaded adata_ref shape: {adata_ref.shape}")

# Find genes common to both datasets initially
common_genes_initial = list(set(adata_st.var_names) & set(adata_ref.var_names))
common_genes_initial.sort()

print(f"\nFound {len(common_genes_initial)} common genes initially across both datasets.")

# Subset both adata_st and adata_ref to this common set
adata_st = adata_st[:, common_genes_initial].copy()
adata_ref = adata_ref[:, common_genes_initial].copy()

print(f"adata_st shape after initial common gene subsetting: {adata_st.shape}")
print(f"adata_ref shape after initial common gene subsetting: {adata_ref.shape}")


####################################################################################################
# --- Step 2: Preprocessing and Filtering for adata_ref (for RegressionModel input) ----------------
####################################################################################################

print("\n--- Step 2: Preprocessing and Filtering adata_ref for reference model ---")

# Store raw counts for adata_ref *after* initial gene subsetting.
if not hasattr(adata_ref, 'raw') or adata_ref.raw is None or adata_ref.raw.shape != adata_ref.shape:
    adata_ref.raw = adata_ref.copy()
    print("adata_ref.raw created/updated with raw counts for reference model input.")
else:
    print("adata_ref.raw already exists.")


reference_label_key = choose_reference_obs_key(
    adata_ref.obs,
    [
        "nanostring_reference",
        "cell_type",
        "annotation.l2",
        "subclass.l2",
        "annotation.l3",
        "subclass.full",
        "annotation.l1",
        "subclass.l1",
    ],
    "label",
)
reference_batch_key = choose_reference_obs_key(
    adata_ref.obs,
    [
        "original_sample_id",
        "library",
        "patient",
        "experiment",
        "specimen",
    ],
    "batch",
)
print(f"Using reference label column: {reference_label_key}")
print(f"Using reference batch column: {reference_batch_key}")

# Filter out cells with missing labels
adata_ref = adata_ref[adata_ref.obs[reference_label_key].notnull(), :].copy()
if adata_ref.n_obs == 0:
    raise ValueError("No annotated cells remaining in adata_ref after filtering NaNs for labels. Cannot train model.")
adata_ref.obs[reference_label_key] = adata_ref.obs[reference_label_key].astype('category')
adata_ref.obs[reference_batch_key] = adata_ref.obs[reference_batch_key].astype(str)
print(f"adata_ref shape after NaN filtering: {adata_ref.shape}")

# Filter genes in adata_ref using cell2location's filtering utility.
# This line generates the plot!
selected_genes_for_model = c2l.utils.filtering.filter_genes(
    adata_ref,
    cell_count_cutoff=5,
    cell_percentage_cutoff2=0.03,
    nonz_mean_cutoff=1.12,
    # Add this parameter to prevent the plot from showing immediately
    # and allow saving it explicitly
)
print(f"Number of genes selected by c2l.utils.filtering.filter_genes: {len(selected_genes_for_model)}")

# --- Add lines to save the figure here ---
output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs_3000epochs_500samples_NMF-k4"
os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

figure_path = os.path.join(output_dir, "gene_filter_accuracy_plot.png") # Choose a descriptive name
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free up memory

print(f"Gene filter plot saved to: {figure_path}")
# --- End of added lines ---


# IMPORTANT: For RegressionModel, adata_ref.X must be RAW (integer) counts.
if adata_ref.raw is not None:
    adata_ref.X = adata_ref.raw.X.copy()
    if isinstance(adata_ref.X, (scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix)):
        adata_ref.X = adata_ref.X.toarray()
    if not np.issubdtype(adata_ref.X.dtype, np.integer):
        adata_ref.X = adata_ref.X.astype(np.int32)
    print("adata_ref.X ensured to be raw (integer) counts for RegressionModel input.")
else:
    print("WARNING: adata_ref.raw is not available. Ensure adata_ref.X contains raw integer counts.")


####################################################################################################
# --- Step 3: FINAL GENE ALIGNMENT for BOTH adata_st and adata_ref ---------------------------------
####################################################################################################

print("\n--- Step 3: Performing final gene alignment for both adata_st and adata_ref ---")

adata_st = adata_st[:, selected_genes_for_model].copy()
adata_ref = adata_ref[:, selected_genes_for_model].copy()

adata_st.var_names_make_unique()
adata_ref.var_names_make_unique()

print(f"FINAL adata_st shape after all gene alignment: {adata_st.shape}")
print(f"FINAL adata_ref shape after all gene alignment: {adata_ref.shape}")
assert np.all(adata_st.var_names == adata_ref.var_names), "FATAL ERROR: Gene names and order DO NOT MATCH after final alignment!"
print("Gene sets for adata_st and adata_ref are now perfectly aligned.")


####################################################################################################
# --- Step 4: Set up and Train the Cell2location RegressionModel (for reference) -------------------
####################################################################################################

print("\n--- Step 4: Setting up and Training Cell2location RegressionModel (for reference) ---")

c2l.models.RegressionModel.setup_anndata(
    adata=adata_ref,
    batch_key=reference_batch_key,
    labels_key=reference_label_key,
    categorical_covariate_keys=[],
    layer=None,
)
print("AnnData setup complete for RegressionModel.")

N_CELL_TYPES = len(adata_ref.obs[reference_label_key].cat.categories)
model_ref_trained = c2l.models.RegressionModel(
    adata_ref
)
print(f"Number of cell types (N_CELL_TYPES): {N_CELL_TYPES}")
print(f"Number of genes (N_GENES): {adata_ref.n_vars}")
print("Reference Model initialized.")

model_ref_trained.train(
    max_epochs=500,
    batch_size=2500,
    train_size=1,
    lr=0.002,
    accelerator='cpu',
)
print("\nReference Model training complete.")
model_ref_trained.save(ref_model_path, overwrite=True)
print(f"\nReference Model saved to: {ref_model_path}")

# Plotting the history
model_ref_trained.plot_history(20) # Use model_ref_trained here

# --- Add lines to save the figure here ---
output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs_3000epochs_500samples_NMF-k4"
os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

figure_path = os.path.join(output_dir, "ref_model_training_history.png") # Choose a descriptive name
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free up memory

print(f"Reference model training history plot saved to: {figure_path}")

####################################################################################################
# --- Step 5: Export posterior and get signatures (inf_aver) ---------------------------------------
####################################################################################################

print("\n--- Step 5: Exporting posterior and getting cell type signatures (inf_aver) ---")

model_ref_trained.export_posterior(
    adata_ref,
    sample_kwargs={
        "num_samples": 1000,
        "batch_size": 2500,
    },
)

# --- Add this line to plot QC and then save it ---
# Call plot_QC() on your trained model
model_ref_trained.plot_QC()

# Define output directory (assuming ref_model_dir is already defined as in your full script)
output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs_3000epochs_500samples_NMF-k4"
os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

# Save the QC plot
qc_figure_path = os.path.join(output_dir, "ref_model_QC_plot.png") # Choose a descriptive name
plt.savefig(qc_figure_path, dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free up memory

print(f"\nReference model QC plot saved to: {qc_figure_path}")

if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
    inf_aver_raw = pd.DataFrame(
        adata_ref.varm["means_per_cluster_mu_fg"],
        index=adata_ref.var_names
    )
    
    cell_type_names = adata_ref.obs[reference_label_key].cat.categories.tolist()
    
    selected_cols_for_inf_aver = []
    for col in inf_aver_raw.columns:
        for cell_type in cell_type_names:
            if col == f"means_per_cluster_mu_fg_{cell_type}" or col == cell_type:
                selected_cols_for_inf_aver.append(col)
                break 
            
    if not selected_cols_for_inf_aver:
        if len(inf_aver_raw.columns) == len(cell_type_names):
            print("Warning: Standard 'means_per_cluster_mu_fg_' prefix not found. Assuming column order matches cell_type_names.")
            selected_cols_for_inf_aver = inf_aver_raw.columns.tolist()
        else:
            raise KeyError("Could not identify cell type columns in 'means_per_cluster_mu_fg'. Please inspect adata_ref.varm['means_per_cluster_mu_fg'].columns to find the correct naming convention.")

    inf_aver = inf_aver_raw[selected_cols_for_inf_aver].copy()
    
    final_columns = []
    for col_name in inf_aver.columns:
        if col_name.startswith("means_per_cluster_mu_fg_"):
            final_columns.append(col_name.replace("means_per_cluster_mu_fg_", ""))
        else:
            final_columns.append(col_name)
    inf_aver.columns = final_columns

    if len(inf_aver.columns) != len(cell_type_names):
        print(f"WARNING: Number of columns in inf_aver ({len(inf_aver.columns)}) does not match number of cell types ({len(cell_type_names)}).")
        print(f"This might indicate an issue with column selection. Please manually inspect inf_aver.columns and adata_ref.obs[{reference_label_key!r}].cat.categories.")

else:
    raise KeyError("Could not find 'means_per_cluster_mu_fg' in adata_ref.varm. Check cell2location version or export_posterior output structure.")

print("\nEstimated cell type signatures (inf_aver) DataFrame created.")
print("inf_aver.shape:", inf_aver.shape)
print("inf_aver.head():")
print(inf_aver.head())

inf_aver_csv_path = os.path.join(ref_model_dir, "inf_aver_3000ep_500samp_NMF-k4.csv")
inf_aver.to_csv(inf_aver_csv_path)
print(f"\ninf_aver saved to: {inf_aver_csv_path}")

#########################################################################################################
# --- Step 6: Prepare adata_st for the Cell2location (spatial) model (including Spatial Coordinates) ----
#########################################################################################################

print("\n--- Step 6: Preparing adata_st for the Cell2location (spatial) model ---")

# IMPORTANT: For the spatial Cell2location model with GammaPoisson likelihood,
# adata_st.X must also contain RAW (integer) counts.
# We are ensuring adata_st.X is subsetted to the selected genes and then set to raw counts.
if adata_st.raw is not None:
    # Subset adata_st.raw to match the genes in current adata_st
    adata_st.X = adata_st.raw[:, adata_st.var_names].X # Use the raw counts for selected genes
    
    # Check if the data is sparse and convert to dense if necessary for type conversion
    if isinstance(adata_st.X, (scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix)):
        adata_st.X = adata_st.X.toarray()
    
    # Ensure data is integer type
    if not np.issubdtype(adata_st.X.dtype, np.integer):
        adata_st.X = adata_st.X.astype(np.int32)
    print("adata_st.X ensured to be raw (integer) counts for spatial model input, and genes aligned.")
else:
    print("WARNING: adata_st.raw is not available. Ensure adata_st.X contains raw integer counts.")


print("\n\n#########################################################################")
print("WARNING: Spatial coordinates (adata_st.obsm['spatial']) are still MISSING.")
print("         This is ABSOLUTELY REQUIRED for the spatial Cell2location model.")
print("         The next step (model initialization/training) WILL FAIL without these.")
print("#########################################################################")

spatial_coords_present = False
if 'unique_cell_id' in adata_st.obs.columns:
    adata_st.obs['unique_cell_id'] = adata_st.obs['unique_cell_id'].astype(str)
    adata_st.obs_names = adata_st.obs['unique_cell_id']
    adata_st.obs_names_make_unique()
    print(f"adata_st.obs_names reused from existing unique_cell_id. Example: {adata_st.obs_names[0]}")
if 'fov' not in adata_st.obs.columns or 'cell_ID' not in adata_st.obs.columns:
    print("ERROR: 'fov' or 'cell_ID' columns not found in adata_st.obs. Cannot create unique cell IDs for spatial alignment.")
    print("Please ensure your initial adata_st loading includes these columns in .obs.")
else:
    if 'unique_cell_id' not in adata_st.obs.columns:
        if 'patient' in adata_st.obs.columns:
            adata_st.obs['unique_cell_id'] = (
                adata_st.obs['patient'].astype(str) + '_' +
                adata_st.obs['fov'].astype(str) + '_' +
                adata_st.obs['cell_ID'].astype(str)
            )
            spatial_id_mode = "patient_fov_cell_ID"
        else:
            adata_st.obs['unique_cell_id'] = adata_st.obs['fov'].astype(str) + '_' + adata_st.obs['cell_ID'].astype(str)
            spatial_id_mode = "fov_cell_ID"
        adata_st.obs_names = adata_st.obs['unique_cell_id']
        adata_st.obs_names_make_unique()

        print(f"adata_st.obs_names recreated as unique_cell_id ({spatial_id_mode}). Example: {adata_st.obs_names[0]}")

    cell_metadata_file_name = "GSE234713_CosMx_cell_metadata.csv.gz" 
    spatial_metadata_path = os.path.join("/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/", cell_metadata_file_name)

    if not os.path.exists(spatial_metadata_path):
        print(f"\nERROR: Spatial metadata file not found at '{spatial_metadata_path}'.")
        print("Please confirm the exact filename and path of your 'Cell Metadata File' and update 'cell_metadata_file_name' in the code.")
    else:
        print(f"\nFound spatial metadata file: {spatial_metadata_path}. Loading...")
        metadata_compression = 'gzip' if spatial_metadata_path.endswith('.gz') else None
        spatial_df = pd.read_csv(spatial_metadata_path, compression=metadata_compression)

        if 'unique_cell_id' in spatial_df.columns:
            spatial_df['unique_cell_id'] = spatial_df['unique_cell_id'].astype(str)
        elif 'patient' in spatial_df.columns and 'fov' in spatial_df.columns and 'cell_ID' in spatial_df.columns:
            spatial_df['unique_cell_id'] = (
                spatial_df['patient'].astype(str) + '_' +
                spatial_df['fov'].astype(str) + '_' +
                spatial_df['cell_ID'].astype(str)
            )
        elif 'fov' in spatial_df.columns and 'cell_ID' in spatial_df.columns:
            spatial_df['unique_cell_id'] = spatial_df['fov'].astype(str) + '_' + spatial_df['cell_ID'].astype(str)

        if 'unique_cell_id' in spatial_df.columns:
            spatial_df.set_index('unique_cell_id', inplace=True)
            
            common_cells_st_spatial = list(set(adata_st.obs_names) & set(spatial_df.index))
            if len(common_cells_st_spatial) == 0:
                print("ERROR: No common cell IDs found between adata_st and spatial metadata file after creating unique_cell_id. Cannot align spatial coordinates.")
                spatial_coords_present = False
            else:
                spatial_df = spatial_df.loc[adata_st.obs_names].copy()
                
                if 'CenterX_global_px' in spatial_df.columns and 'CenterY_global_px' in spatial_df.columns:
                    adata_st.obsm['spatial'] = spatial_df[['CenterX_global_px', 'CenterY_global_px']].values
                    print("Spatial coordinates loaded and added to adata_st.obsm['spatial'].")
                    spatial_coords_present = True
                else:
                    print("ERROR: 'CenterX_global_px' or 'CenterY_global_px' columns not found in spatial metadata file. Cannot add spatial coordinates.")
                    spatial_coords_present = False
        else:
            print("ERROR: 'fov' or 'cell_ID' columns not found in the spatial metadata file. Cannot create unique cell IDs for alignment.")
            spatial_coords_present = False

if not spatial_coords_present:
    print("Final check: adata_st.obsm['spatial'] is still NOT populated. Please resolve the issue above.")
else:
    print("Spatial coordinates successfully added. You are ready to initialize and setup the spatial model.")


c2l.models.Cell2location.setup_anndata(
    adata=adata_st,
    batch_key="patient",
)

model = c2l.models.Cell2location(
    adata_st,
    cell_state_df=inf_aver,
    N_cells_per_location=1,
)
model.view_anndata_setup()

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

model.train(max_epochs=3000, batch_size=None, train_size=1, accelerator='cpu')

# plot training history
model.plot_history()

output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs_3000epochs_500samples_NMF-k4"
os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

figure_path = os.path.join(output_dir, "spatial_model_training_history.png") # Choose a descriptive name
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free up memory

print(f"Spatial model training history plot saved to: {figure_path}")

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

adata_st = model.export_posterior(
    adata_st,
    sample_kwargs={
        "num_samples": 500,
        "batch_size": math.ceil(model.adata.n_obs / 50),
        "accelerator": "cpu",
    },
)

# Plotting the QC
model.plot_QC()

# --- Add lines to save the figure here ---
output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs_3000epochs_500samples_NMF-k4"
os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

figure_path = os.path.join(output_dir, "spatial_model_QC_plot.png") # Choose a descriptive name
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free up memory

print(f"Spatial model QC plot saved to: {figure_path}")

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# --- Code to save inferred cell type abundances ---
print("\n# Save inferred cell type proportions (cells x cell types)")
try:
    # Extract the inferred cell type abundances from adata_st.uns
    # Based on your screenshot, this is where the cell type means are stored
    cell_abundance_df = pd.DataFrame(
        adata_st.uns["mod"]["post_sample_means"]["w_sf"],
        index=adata_st.obs_names,
        columns=adata_st.uns["mod"]["factor_names"]
    )

    # Define the path for the CSV file in your Outputs folder
    abundance_csv_path = os.path.join(output_dir, "inferred_cell_type_abundances.csv")
    cell_abundance_df.to_csv(abundance_csv_path)

    print(f"Inferred cell type abundances saved to: {abundance_csv_path}")
    print("? Inferred cell type abundances saved.")

except KeyError as e:
    print(f"ERROR: Could not find expected keys for cell abundance data. {e}")
    print("Please check the structure of adata_st.uns['mod'] after export_posterior.")
    print("Keys in adata_st.uns['mod']: ", adata_st.uns['mod'].keys())
    # Optionally, you can add more debug prints to inspect the structure if error persists
except Exception as e:
    print(f"An unexpected error occurred while saving cell abundances: {e}")
# --- End of code to save inferred cell type abundances ---


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# --- NMF Analysis and Output Saving ---
print("\n--- Starting NMF Analysis ---")

# Define the output directory for NMF results.
nmf_output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs_3000epochs_500samples_NMF-k4"
os.makedirs(nmf_output_dir, exist_ok=True)
print(f"NMF outputs will be saved to: {nmf_output_dir}")

# PREREQUISITE CHECK:
try:
    # Extract abundance matrix (cells × cell types) from adata_st.
    # The np.array() call ensures it's a dense matrix for NMF.
    X = np.array(adata_st.uns["mod"]["post_sample_means"]["w_sf"])
    print(f"✔ Extracted abundance matrix (X) for NMF: {X.shape}")
except NameError:
    print("ERROR: 'adata_st' object is not defined. Please ensure previous steps have run.")
    exit()
except KeyError as e:
    print(f"ERROR: Expected key '{e}' not found in adata_st.uns for NMF input.")
    exit()

# --- Run NMF with a Defined k ---
# Define the user-set number of factors (k)
n_components = 4
print(f"\n--- Performing NMF with a fixed k={n_components} components ---")

# Initialize and fit the NMF model
nmf = NMF(n_components=n_components, init='nndsvda', random_state=0, max_iter=1000)
W = nmf.fit_transform(X)  # cells × factors
H = nmf.components_       # factors × cell types

print("✔ NMF factorization complete.")
print(f"Shape of W matrix (cells x factors): {W.shape}")
print(f"Shape of H matrix (factors x cell types): {H.shape}")

# Save final W and H matrices
w_matrix_path = os.path.join(nmf_output_dir, "NMF_W_matrix.npy")
h_matrix_path = os.path.join(nmf_output_dir, "NMF_H_matrix.npy")
np.save(w_matrix_path, W)
np.save(h_matrix_path, H)
print(f"Final W matrix saved to: {w_matrix_path}")
print(f"Final H matrix saved to: {h_matrix_path}")

# Assign dominant NMF factor per cell
new_nmf_column_name = 'dominant_nmf_factor'
adata_st.obs[new_nmf_column_name] = pd.Series(np.argmax(W, axis=1), index=adata_st.obs.index)
adata_st.obs[new_nmf_column_name] = adata_st.obs[new_nmf_column_name].astype('category')
print(f"✔ NMF dominant factor assigned to adata_st.obs['{new_nmf_column_name}'].")


# --- Plot and save NMF factor distribution ---
print("\n--- Plotting NMF factor distribution ---")
plt.figure(figsize=(8, 6))
# Note: We use `new_nmf_column_name` to ensure we use the correct column
sns.countplot(x=new_nmf_column_name, data=adata_st.obs, palette="viridis", order=sorted(adata_st.obs[new_nmf_column_name].unique()))
plt.title("Cell Counts per NMF-Inferred Niche")
plt.xlabel("NMF Factor")
plt.ylabel("Number of Cells")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
factor_dist_path = os.path.join(nmf_output_dir, "NMF_Factor_Distribution_Plot.png")
plt.savefig(factor_dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"NMF factor distribution plot saved to: {factor_dist_path}")
# --- End of plot section ---


# Create 'cell_type' column for crosstab if it doesn't exist
if "cell_type" not in adata_st.obs.columns:
    print("\nINFO: 'cell_type' column not found. Creating it from dominant inferred cell types.")
    if 'cell_abundance_df' in locals():
        dominant_cell_types = cell_abundance_df.idxmax(axis=1)
        adata_st.obs["cell_type"] = dominant_cell_types.reindex(adata_st.obs.index)
        adata_st.obs["cell_type"] = adata_st.obs["cell_type"].astype('category')
        print("✔ 'cell_type' column created in adata_st.obs.")
    else:
        print("ERROR: 'cell_abundance_df' not found. Cannot create 'cell_type' column.")

# Tabulate: NMF Factor × Dominant Cell Type
if "cell_type" in adata_st.obs.columns:
    print(f"\nCalculating crosstab for '{new_nmf_column_name}' vs 'cell_type'...")
    niche_celltype = pd.crosstab(adata_st.obs[new_nmf_column_name], adata_st.obs["cell_type"])
    
    if not niche_celltype.empty:
        # Normalize to get proportions
        print("Normalizing crosstab counts to proportions...")
        niche_sums = niche_celltype.sum(axis=1)
        niche_celltype_norm = niche_celltype.div(niche_sums + 1e-9, axis=0).fillna(0)

        # Save the normalized proportions to CSV
        niche_celltype_norm_path = os.path.join(nmf_output_dir, "NMF_Niche_CellType_Proportions_Normalized.csv")
        niche_celltype_norm.to_csv(niche_celltype_norm_path)
        print(f"Normalized NMF niche-celltype proportions saved to: {niche_celltype_norm_path}")
    else:
        print("WARNING: Crosstab resulted in an empty DataFrame.")
else:
    print("NMF niche-celltype proportions table could not be generated as 'cell_type' column is missing.")

print("\n--- NMF Analysis Complete ---")
