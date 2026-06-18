#!/usr/bin/env python3
# Canonical stage template: this file is intended to be patched and launched by
# run_pipeline.py as the CPU NMF follow-up after a cell2location stage.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from kneed import KneeLocator
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

input_h5ad_path = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs/cosmx_cell2loc_only.h5ad"
nmf_output_dir = "/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs"
os.makedirs(nmf_output_dir, exist_ok=True)
nmf_h5ad_path = os.path.join(nmf_output_dir, "cosmx_with_nmf.h5ad")

nmf_selection_method = "elbow_k"
K_range = range(2, 21)
poisson_n_runs = 10
poisson_max_iter = 5000
poisson_normalize_rows_to_sum1 = False
poisson_cumulative_improvement_target = 0.95
poisson_eps = 1e-8
nmf_backend = "sklearn"
nmf_device = "auto"


class NMFResult:
    def __init__(self, reconstruction_err_):
        self.reconstruction_err_ = float(reconstruction_err_)


def _resolve_torch_device(device_name):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch backend requested for NMF, but torch is not installed.")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for NMF, but torch.cuda.is_available() is false.")
    return device


def _torch_dtype_for_device(device):
    return torch.float32 if device.type == "cuda" else torch.float64


def _torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _torch_factor_redundancy(H):
    H_norm = torch.linalg.norm(H, dim=1, keepdim=True).clamp_min(poisson_eps)
    sim = (H / H_norm) @ (H / H_norm).T
    sim.fill_diagonal_(0.0)
    return float(torch.max(sim).detach().cpu().item())


def _torch_frobenius_error(X, W, H):
    return float(torch.linalg.norm(X - W @ H, ord="fro").detach().cpu().item())


def _torch_kl_error(X, WH):
    ratio = X / WH.clamp_min(poisson_eps)
    value = X * torch.log(ratio.clamp_min(poisson_eps)) - X + WH
    return float(torch.sum(value).detach().cpu().item())


def _torch_init_factors(X, k, seed, device):
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    dtype = _torch_dtype_for_device(device)
    n_samples, n_features = X.shape
    W = torch.rand((n_samples, k), generator=generator, device=device, dtype=dtype).clamp_min(1e-4)
    H = torch.rand((k, n_features), generator=generator, device=device, dtype=dtype).clamp_min(1e-4)
    return W, H


def _run_torch_nmf(matrix, k, seed, max_iter, loss):
    device = _resolve_torch_device(nmf_device)
    dtype = _torch_dtype_for_device(device)
    X_t = torch.as_tensor(matrix, device=device, dtype=dtype).clamp_min(poisson_eps)
    W, H = _torch_init_factors(X_t, k, seed, device)

    previous_error = None
    patience = 25
    stale_steps = 0

    for iteration in range(int(max_iter)):
        WH = (W @ H).clamp_min(poisson_eps)
        if loss == "frobenius":
            H = H * ((W.T @ X_t) / ((W.T @ W @ H).clamp_min(poisson_eps)))
            WH = (W @ H).clamp_min(poisson_eps)
            W = W * ((X_t @ H.T) / ((W @ (H @ H.T)).clamp_min(poisson_eps)))
        elif loss == "kullback-leibler":
            ratio = X_t / WH
            H = H * ((W.T @ ratio) / (torch.sum(W, dim=0, keepdim=True).T.clamp_min(poisson_eps)))
            WH = (W @ H).clamp_min(poisson_eps)
            ratio = X_t / WH
            W = W * ((ratio @ H.T) / (torch.sum(H, dim=1, keepdim=True).T.clamp_min(poisson_eps)))
        else:
            raise ValueError(f"Unsupported torch NMF loss: {loss}")

        W = W.clamp_min(poisson_eps)
        H = H.clamp_min(poisson_eps)

        if iteration % 20 == 0 or iteration == int(max_iter) - 1:
            if loss == "frobenius":
                current_error = _torch_frobenius_error(X_t, W, H)
            else:
                current_error = _torch_kl_error(X_t, (W @ H).clamp_min(poisson_eps))
            if previous_error is not None and abs(previous_error - current_error) <= 1e-5 * max(1.0, previous_error):
                stale_steps += 1
                if stale_steps >= patience:
                    break
            else:
                stale_steps = 0
            previous_error = current_error

    if previous_error is None:
        if loss == "frobenius":
            previous_error = _torch_frobenius_error(X_t, W, H)
        else:
            previous_error = _torch_kl_error(X_t, (W @ H).clamp_min(poisson_eps))

    return _torch_to_numpy(W), _torch_to_numpy(H), NMFResult(previous_error)


def run_standard_nmf(matrix, k, seed=0, max_iter=1000):
    if nmf_backend == "torch":
        return _run_torch_nmf(matrix, k, seed, max_iter=max_iter, loss="frobenius")
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=max_iter)
    W = model.fit_transform(matrix)
    H = model.components_
    return W, H, model


def run_kl_nmf(matrix, k, seed, max_iter=5000):
    if nmf_backend == "torch":
        return _run_torch_nmf(matrix, k, seed, max_iter=max_iter, loss="kullback-leibler")
    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        tol=1e-4,
        random_state=seed,
    )
    W = model.fit_transform(matrix)
    H = model.components_
    return W, H, model


def factor_redundancy(H):
    if nmf_backend == "torch":
        device = _resolve_torch_device(nmf_device)
        H_t = torch.as_tensor(H, device=device, dtype=_torch_dtype_for_device(device))
        return _torch_factor_redundancy(H_t)
    sim = cosine_similarity(H)
    np.fill_diagonal(sim, 0.0)
    return float(np.max(sim))


print("--- Loading cell2location-only h5ad for NMF ---")
adata_st = sc.read(input_h5ad_path)
print(f"Loaded adata_st shape: {adata_st.shape}")
print(f"NMF backend: {nmf_backend}")
print(f"NMF device: {nmf_device}")
if TORCH_AVAILABLE:
    print(f"Torch CUDA available: {torch.cuda.is_available()}")

try:
    X = np.array(adata_st.uns["mod"]["post_sample_means"]["w_sf"])
    print(f"Extracted abundance matrix (X) for NMF: {X.shape}")
except KeyError as exc:
    raise KeyError(f"Expected key {exc!s} not found in adata_st.uns for NMF input.") from exc

optimal_k = None
W = None
H = None
nmf = None

if nmf_selection_method == "fixed_k":
    fixed_k = list(K_range)[0]
    print(f"\n--- Performing fixed-rank NMF with k={fixed_k} components ---")
    W, H, nmf = run_standard_nmf(X, fixed_k, seed=0, max_iter=1000)
    optimal_k = fixed_k
elif nmf_selection_method in ("poisson_redundancy_k", "poisson_cumulative_improvement_k"):
    if nmf_selection_method == "poisson_redundancy_k":
        print("\n--- Determining optimal number of NMF factors (k) using Poisson/KL redundancy ---")
    else:
        print("\n--- Determining optimal number of NMF factors (k) using Poisson/KL cumulative reconstruction improvement ---")
    print("Testing k in range:", list(K_range))

    X_poisson = np.clip(np.array(X, dtype=np.float64), 0, None) + poisson_eps
    if poisson_normalize_rows_to_sum1:
        X_poisson = X_poisson / (X_poisson.sum(axis=1, keepdims=True) + poisson_eps)

    selection_rows = []
    for k in tqdm(K_range, desc="Running Poisson/KL-NMF for different k"):
        redundancies = []
        reconstruction_errors = []
        best_seed = None
        best_redundancy = np.inf

        for seed in tqdm(range(poisson_n_runs), desc=f"Runs for k={k}", leave=False):
            try:
                _W, _H, _model = run_kl_nmf(X_poisson, k, seed, max_iter=poisson_max_iter)
                redundancy = factor_redundancy(_H)
                redundancies.append(redundancy)
                reconstruction_errors.append(float(_model.reconstruction_err_))
                if redundancy < best_redundancy:
                    best_redundancy = redundancy
                    best_seed = seed
            except Exception as exc:
                print(f"[WARNING] Poisson/KL-NMF failed for k={k}, seed={seed}: {exc}")

        if redundancies:
            selection_rows.append(
                {
                    "K": int(k),
                    "redundancy_mean": float(np.mean(redundancies)),
                    "redundancy_sd": float(np.std(redundancies)),
                    "redundancy_min": float(np.min(redundancies)),
                    "best_seed": int(best_seed) if best_seed is not None else np.nan,
                    "reconstruction_error_mean": float(np.mean(reconstruction_errors)),
                    "reconstruction_error_sd": float(np.std(reconstruction_errors)),
                    "n_successful_runs": int(len(redundancies)),
                }
            )
        else:
            selection_rows.append(
                {
                    "K": int(k),
                    "redundancy_mean": np.nan,
                    "redundancy_sd": np.nan,
                    "redundancy_min": np.nan,
                    "best_seed": np.nan,
                    "reconstruction_error_mean": np.nan,
                    "reconstruction_error_sd": np.nan,
                    "n_successful_runs": 0,
                }
            )

    selection_df = pd.DataFrame(selection_rows).sort_values("K")
    if "reconstruction_error_mean" in selection_df.columns:
        valid_recon = selection_df["reconstruction_error_mean"].dropna()
        if not valid_recon.empty:
            E_start = valid_recon.iloc[0]
            E_end = valid_recon.iloc[-1]
            denom = E_start - E_end
            if abs(denom) < 1e-12:
                selection_df["fraction_total_improvement"] = np.nan
            else:
                selection_df["fraction_total_improvement"] = (
                    (E_start - selection_df["reconstruction_error_mean"]) / denom
                )

    selection_csv_path = os.path.join(nmf_output_dir, "NMF_Poisson_Redundancy_By_K.csv")
    selection_df.to_csv(selection_csv_path, index=False)
    print(f"Poisson/KL redundancy summary saved to: {selection_csv_path}")

    valid_df = selection_df.dropna(subset=["redundancy_mean"])
    if valid_df.empty:
        raise ValueError("Poisson/KL-NMF rank selection failed for every tested K.")

    if nmf_selection_method == "poisson_redundancy_k":
        best_row = valid_df.sort_values(["redundancy_mean", "K"]).iloc[0]
    else:
        if "fraction_total_improvement" not in valid_df.columns:
            raise ValueError("Missing fraction_total_improvement for Poisson cumulative improvement selection.")
        improving_df = valid_df.dropna(subset=["fraction_total_improvement"])
        if improving_df.empty:
            raise ValueError("Poisson cumulative improvement selection failed because no reconstruction error values were available.")
        hits = improving_df.loc[
            improving_df["fraction_total_improvement"] >= poisson_cumulative_improvement_target
        ]
        if hits.empty:
            best_row = improving_df.sort_values("K").iloc[-1]
            print(
                f"WARNING: No K reached the cumulative improvement target of {poisson_cumulative_improvement_target:.3f}. "
                f"Defaulting to the largest tested K={int(best_row['K'])}."
            )
        else:
            best_row = hits.sort_values("K").iloc[0]

    optimal_k = int(best_row["K"])
    best_seed = int(best_row["best_seed"])

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        valid_df["K"],
        valid_df["redundancy_mean"],
        yerr=valid_df["redundancy_sd"],
        marker="o",
        capsize=4,
    )
    plt.axvline(optimal_k, linestyle="--", linewidth=1.5, label=f"Selected K={optimal_k}")
    plt.title("Poisson/KL-NMF redundancy vs number of factors")
    plt.xlabel("Number of factors (k)")
    plt.ylabel("Mean factor redundancy")
    plt.xticks(list(K_range))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    redundancy_plot_path = os.path.join(nmf_output_dir, "NMF_Poisson_Redundancy_vs_K.png")
    plt.savefig(redundancy_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Poisson/KL redundancy plot saved to: {redundancy_plot_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(valid_df["K"], valid_df["reconstruction_error_mean"], marker="o")
    plt.axvline(optimal_k, linestyle="--", linewidth=1.5, label=f"Selected K={optimal_k}")
    plt.title("Poisson/KL-NMF reconstruction error vs number of factors")
    plt.xlabel("Number of factors (k)")
    plt.ylabel("Mean reconstruction error")
    plt.xticks(list(K_range))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    recon_plot_path = os.path.join(nmf_output_dir, "NMF_Poisson_Reconstruction_Error_vs_K.png")
    plt.savefig(recon_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Poisson/KL reconstruction error plot saved to: {recon_plot_path}")

    if "fraction_total_improvement" in valid_df.columns and not valid_df["fraction_total_improvement"].isna().all():
        plt.figure(figsize=(10, 6))
        plt.plot(valid_df["K"], valid_df["fraction_total_improvement"], marker="o")
        if nmf_selection_method == "poisson_cumulative_improvement_k":
            plt.axhline(
                poisson_cumulative_improvement_target,
                linestyle="--",
                linewidth=1.5,
                color="tab:orange",
                label=f"Target={poisson_cumulative_improvement_target:.2f}",
            )
        plt.axvline(optimal_k, linestyle="--", linewidth=1.5, label=f"Selected K={optimal_k}")
        plt.title("Poisson/KL-NMF cumulative reconstruction improvement vs number of factors")
        plt.xlabel("Number of factors (k)")
        plt.ylabel("Fraction of total improvement achieved")
        plt.xticks(list(K_range))
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        improvement_plot_path = os.path.join(nmf_output_dir, "NMF_Poisson_Cumulative_Improvement_vs_K.png")
        plt.savefig(improvement_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Poisson/KL cumulative improvement plot saved to: {improvement_plot_path}")

    diagnostics_path = os.path.join(nmf_output_dir, "NMF_Poisson_Selected_K.txt")
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        handle.write(f"selection_method: {nmf_selection_method}\n")
        if nmf_selection_method == "poisson_cumulative_improvement_k":
            handle.write(f"cumulative_improvement_target: {poisson_cumulative_improvement_target}\n")
        handle.write(f"selected_k: {optimal_k}\n")
        handle.write(f"best_seed: {best_seed}\n")
        handle.write(best_row.to_string())
    print(f"Selected-K diagnostics saved to: {diagnostics_path}")

    print(f"\n--- Performing final Poisson/KL-NMF with k={optimal_k} components and seed={best_seed} ---")
    W, H, nmf = run_kl_nmf(X_poisson, optimal_k, best_seed, max_iter=poisson_max_iter)
else:
    print("\n--- Determining optimal number of NMF factors (k) using the Elbow Method ---")
    reconstruction_errors = []

    print("Testing k in range:", list(K_range))
    for k in tqdm(K_range, desc="Running NMF for different k"):
        _W, _H, _model = run_standard_nmf(X, k, seed=0, max_iter=500)
        reconstruction_errors.append(_model.reconstruction_err_)

    kneedle = KneeLocator(K_range, reconstruction_errors, S=1.0, curve="convex", direction="decreasing")
    optimal_k = kneedle.elbow

    if optimal_k is None:
        print("\nWARNING: Could not automatically find elbow. Defaulting to k=12.")
        optimal_k = 12
    else:
        print(f"\nOptimal number of factors (k) found at: {optimal_k}")

    plt.figure(figsize=(10, 6))
    kneedle.plot_knee()
    plt.title("NMF Elbow Method for Optimal k")
    plt.xlabel("Number of factors (k)")
    plt.ylabel("Reconstruction Error")
    plt.xticks(K_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    elbow_plot_path = os.path.join(nmf_output_dir, "NMF_Elbow_Plot.png")
    plt.savefig(elbow_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Elbow plot saved to: {elbow_plot_path}")

    print(f"\n--- Performing final NMF with k={optimal_k} components ---")
    W, H, nmf = run_standard_nmf(X, optimal_k, seed=0, max_iter=1000)

print("Final NMF factorization complete.")
print(f"Shape of W matrix (cells x factors): {W.shape}")
print(f"Shape of H matrix (factors x cell types): {H.shape}")

w_matrix_path = os.path.join(nmf_output_dir, "NMF_W_matrix.npy")
h_matrix_path = os.path.join(nmf_output_dir, "NMF_H_matrix.npy")
np.save(w_matrix_path, W)
np.save(h_matrix_path, H)
print(f"Final W matrix saved to: {w_matrix_path}")
print(f"Final H matrix saved to: {h_matrix_path}")

new_nmf_column_name = "dominant_nmf_factor"
adata_st.obs[new_nmf_column_name] = pd.Series(np.argmax(W, axis=1), index=adata_st.obs.index)
adata_st.obs[new_nmf_column_name] = adata_st.obs[new_nmf_column_name].astype("category")
print(f"Assigned dominant NMF factor to adata_st.obs['{new_nmf_column_name}'].")
adata_st.obs["NMF_factor"] = adata_st.obs[new_nmf_column_name].astype(int)
adata_st.obs["NMF_factor"] = adata_st.obs["NMF_factor"].astype("category")

plt.figure(figsize=(8, 6))
sns.countplot(
    x=new_nmf_column_name,
    data=adata_st.obs,
    palette="viridis",
    order=sorted(adata_st.obs[new_nmf_column_name].unique()),
)
plt.title("Cell Counts per NMF-Inferred Niche")
plt.xlabel("NMF Factor")
plt.ylabel("Number of Cells")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
factor_dist_path = os.path.join(nmf_output_dir, "NMF_Factor_Distribution_Plot.png")
plt.savefig(factor_dist_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"NMF factor distribution plot saved to: {factor_dist_path}")

cell_abundance_df = pd.DataFrame(
    adata_st.uns["mod"]["post_sample_means"]["w_sf"],
    index=adata_st.obs_names,
    columns=adata_st.uns["mod"]["factor_names"],
)
if "cell_type" not in adata_st.obs.columns:
    print("\nINFO: 'cell_type' column not found. Creating it from dominant inferred cell types.")
    dominant_cell_types = cell_abundance_df.idxmax(axis=1)
    adata_st.obs["cell_type"] = dominant_cell_types.reindex(adata_st.obs.index)
    adata_st.obs["cell_type"] = adata_st.obs["cell_type"].astype("category")
    print("Created 'cell_type' column in adata_st.obs.")

if "cell_type" in adata_st.obs.columns:
    print(f"\nCalculating crosstab for '{new_nmf_column_name}' vs 'cell_type'...")
    niche_celltype = pd.crosstab(adata_st.obs[new_nmf_column_name], adata_st.obs["cell_type"])
    if not niche_celltype.empty:
        print("Normalizing crosstab counts to proportions...")
        niche_sums = niche_celltype.sum(axis=1)
        niche_celltype_norm = niche_celltype.div(niche_sums + 1e-9, axis=0).fillna(0)
        niche_celltype_norm_path = os.path.join(nmf_output_dir, "NMF_Niche_CellType_Proportions_Normalized.csv")
        niche_celltype_norm.to_csv(niche_celltype_norm_path)
        print(f"Normalized NMF niche-celltype proportions saved to: {niche_celltype_norm_path}")
    else:
        print("WARNING: Crosstab resulted in an empty DataFrame.")
else:
    print("NMF niche-celltype proportions table could not be generated as 'cell_type' column is missing.")

for obsm_key in list(adata_st.obsm.keys()):
    obsm_value = adata_st.obsm[obsm_key]
    if isinstance(obsm_value, pd.DataFrame):
        sanitized_columns = pd.Index(obsm_value.columns.astype(str)).str.replace("/", "_", regex=False)
        if not sanitized_columns.equals(obsm_value.columns):
            obsm_value = obsm_value.copy()
            obsm_value.columns = sanitized_columns
            adata_st.obsm[obsm_key] = obsm_value

adata_st.write(nmf_h5ad_path)
print(f"NMF annotated h5ad saved to: {nmf_h5ad_path}")
print("\n--- NMF Analysis Complete ---")
