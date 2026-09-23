#!/usr/bin/env python3
"""Read-only, label-independent spatial validation of frozen NMF and NOVAE domains.

The command is deliberately a SLURM-only workload in normal use: it reads the
calibrated NOVAE H5AD and historical ``post_nmf_obs.csv``, computes both arms on
one complete-case graph/expression contract, and publishes an atomic report.
No disease field is used by this analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

GRAPH_KEY = "spatial_connectivities"
VALID_KEY = "neighborhood_valid"
DOMAIN_KEY = "novae_domains_res1.0"
N_DOMAINS = 9
DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 42
EXPECTED_NOVAE_INPUT_SHA256 = "262418e8e7ed06de805e940406f3ae9e41487ce085da1ae8f940c81f95daf6dd"
EXPECTED_NOVAE_CHECKPOINT_SHA256 = "1422f9f72d6e532921bf8a90f0996f1c46c6891f6ecbc73e404521ec5aa7b04a"
EXPECTED_CALIBRATED_H5AD_SHA256 = "40b60eba32c0716637eae98a28c852c11c53dfe4168570ad3230cf0d43219ffd"
EXPECTED_POST_NMF_OBS_SHA256 = "a79a4e5949752110593f45eccd1ff34786b7d39cea3ce144b635444638bb354b"
EXPECTED_VALID = 13372
EXPECTED_INVALID = 45
EXPECTED_DOMAINS = {f"L{i}" for i in range(N_DOMAINS)}
EXPECTED_NMF_FACTORS = {str(i) for i in range(N_DOMAINS)}


class ContractError(ValueError):
    """A fail-closed biological validation contract error."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _atomic_json(payload: Any, path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(name, index=False)
        else:
            frame.to_csv(name, index=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pick(columns: Iterable[str], choices: tuple[str, ...], label: str) -> str:
    for choice in choices:
        if choice in columns:
            return choice
    raise ContractError(f"missing {label}; expected one of {choices}")


def _text(values: Iterable[Any], label: str) -> np.ndarray:
    result = np.asarray([str(v).strip() for v in values], dtype=object)
    if (result == "").any() or pd.isna(result).any() or np.isin(np.char.lower(result.astype(str)), ["nan", "none", "null"]).any():
        raise ContractError(f"{label} contains missing/blank values")
    return result


def _factor_values(values: Iterable[Any], label: str) -> np.ndarray:
    result = []
    for value in values:
        try:
            number = float(value)
            if np.isfinite(number) and number.is_integer():
                result.append(str(int(number)))
                continue
        except (TypeError, ValueError):
            pass
        result.append(str(value).strip())
    return _text(result, label)


def _bool_values(values: Iterable[Any], label: str) -> np.ndarray:
    result = list(values)
    if any(not isinstance(v, (bool, np.bool_)) for v in result):
        raise ContractError(f"{label} must contain native booleans")
    return np.asarray(result, dtype=bool)


def _finite_coords(adata: Any) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise ContractError("obsm['spatial'] is required")
    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ContractError("obsm['spatial'] must be n_obs by at least two")
    try:
        coords = coords[:, :2].astype(float)
    except (TypeError, ValueError) as exc:
        raise ContractError("spatial coordinates must be numeric") from exc
    if not np.isfinite(coords).all():
        raise ContractError("spatial coordinates contain non-finite values")
    return coords


def _raw_counts(adata: Any) -> Any:
    matrix = adata.X
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    try:
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values), atol=1e-8, rtol=0):
            raise ContractError("X must contain finite nonnegative integer raw counts")
    except TypeError as exc:
        raise ContractError("X must be numeric raw counts") from exc
    if getattr(matrix, "ndim", 2) != 2:
        raise ContractError("X must be two-dimensional")
    return matrix


def validate_graph(graph: Any, slides: Iterable[Any], n_obs: int) -> sparse.csr_matrix:
    """Validate and canonicalize the frozen undirected within-slide graph."""
    if graph is None:
        raise ContractError("spatial_connectivities is missing")
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(np.asarray(graph))
    graph = graph.tocsr().astype(float)
    if graph.shape != (n_obs, n_obs):
        raise ContractError("spatial_connectivities shape does not match observations")
    if not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise ContractError("spatial_connectivities contains invalid weights")
    if (graph - graph.T).nnz:
        raise ContractError("spatial_connectivities must be symmetric")
    if np.any(graph.diagonal() != 0):
        raise ContractError("spatial_connectivities diagonal must be zero")
    slide = np.asarray([str(v) for v in slides])
    rows, cols = graph.nonzero()
    if np.any(slide[rows] != slide[cols]):
        raise ContractError("spatial_connectivities contains cross-slide edges")
    graph.eliminate_zeros()
    return graph


def shared_valid_filter(novae_valid: Iterable[Any], nmf_ids: Iterable[Any], novae_ids: Iterable[Any], *, expected_valid: int | None = None, expected_invalid: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return NOVAE-valid rows shared by both arms, without imputation."""
    left, right = _text(nmf_ids, "NMF unique_cell_id"), _text(novae_ids, "NOVAE unique_cell_id")
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ContractError("unique_cell_id must be unique")
    if set(left) != set(right):
        raise ContractError("NMF and NOVAE unique_cell_id sets differ")
    valid = _bool_values(novae_valid, VALID_KEY)
    if len(valid) != len(right):
        raise ContractError("neighborhood_valid length differs from NOVAE rows")
    mask = valid.copy()
    details = {"total_rows": int(len(valid)), "valid_rows": int(mask.sum()), "invalid_rows": int((~mask).sum()), "imputation": False, "policy": "NOVAE valid rows only; shared complete-case"}
    if expected_valid is not None and details["valid_rows"] != expected_valid:
        raise ContractError(f"expected {expected_valid} valid rows, found {details['valid_rows']}")
    if expected_invalid is not None and details["invalid_rows"] != expected_invalid:
        raise ContractError(f"expected {expected_invalid} invalid rows, found {details['invalid_rows']}")
    return mask, details


def _undirected_edges(graph: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = sparse.triu(graph, k=1).nonzero()
    weights = np.asarray(graph[rows, cols]).ravel()
    return rows, cols, weights


def _within_fraction(labels: np.ndarray, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> float:
    if len(rows) == 0 or weights.sum() <= 0:
        return float("nan")
    return float(weights[labels[rows] == labels[cols]].sum() / weights.sum())


def categorical_assortativity(labels: Iterable[Any], graph: sparse.csr_matrix) -> float:
    labels = np.asarray([str(x) for x in labels])
    rows, cols, weights = _undirected_edges(graph)
    if len(rows) == 0 or weights.sum() <= 0:
        return float("nan")
    categories = sorted(set(labels))
    lookup = {v: i for i, v in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=float)
    for i, j, weight in zip(rows, cols, weights, strict=True):
        a, b = lookup[labels[i]], lookup[labels[j]]
        matrix[a, b] += weight
        matrix[b, a] += weight
    matrix /= matrix.sum()
    trace = np.trace(matrix)
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum())
    return float((trace - expected) / (1.0 - expected)) if expected < 1 else float("nan")


def pas(labels: Iterable[Any], graph: sparse.csr_matrix) -> dict[str, float]:
    """Proportion disagreeing with neighbor majority; ties count as disagreement."""
    values = np.asarray([str(x) for x in labels])
    disagree, eligible, ties, zero_degree = 0, 0, 0, 0
    for index in range(len(values)):
        neighbors = graph.getrow(index).indices
        if len(neighbors) == 0:
            zero_degree += 1
            continue
        counts = pd.Series(values[neighbors]).value_counts()
        top = counts[counts == counts.max()].index.astype(str).tolist()
        eligible += 1
        if len(top) != 1:
            ties += 1
            disagree += 1
        elif values[index] != top[0]:
            disagree += 1
    return {"pas": float(disagree / eligible) if eligible else float("nan"), "eligible_spots": eligible, "tie_spots": ties, "zero_degree_spots": zero_degree}


def fragmentation(labels: Iterable[Any], graph: sparse.csr_matrix, *, spots_scale: float = 100.0) -> pd.DataFrame:
    values = np.asarray([str(x) for x in labels])
    rows: list[dict[str, Any]] = []
    for domain in sorted(set(values)):
        indices = np.flatnonzero(values == domain)
        induced = graph[indices][:, indices]
        components = int(connected_components(induced, directed=False, return_labels=False)) if len(indices) else 0
        component_labels = connected_components(induced, directed=False, return_labels=True)[1] if len(indices) else np.array([], dtype=int)
        sizes = np.bincount(component_labels) if len(component_labels) else np.array([], dtype=int)
        rows.append({"domain": domain, "spots": int(len(indices)), "components": components, "components_per_100_spots": float(components * spots_scale / len(indices)) if len(indices) else float("nan"), "largest_component_fraction": float(sizes.max() / len(indices)) if len(sizes) else float("nan")})
    return pd.DataFrame(rows)


def _permutation_values(labels: np.ndarray, graph: sparse.csr_matrix, slides: np.ndarray, permutations: int, seed: int) -> np.ndarray:
    if permutations < 2:
        raise ContractError("permutations must be at least 2")
    rng = np.random.default_rng(seed)
    rows, cols, weights = _undirected_edges(graph)
    out = np.empty(permutations, dtype=float)
    for p in range(permutations):
        shuffled = labels.copy()
        for slide in sorted(set(slides)):
            indices = np.flatnonzero(slides == slide)
            shuffled[indices] = shuffled[rng.permutation(indices)]
        out[p] = _within_fraction(shuffled, rows, cols, weights)
    return out


def spatial_metrics(labels: Iterable[Any], graph: sparse.csr_matrix, slides: Iterable[Any], *, permutations: int = DEFAULT_PERMUTATIONS, seed: int = DEFAULT_SEED, arm: str = "arm") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-slide spatial metrics and deterministic within-slide nulls."""
    if permutations < 2:
        raise ContractError("permutations must be at least 2")
    labels = np.asarray([str(x) for x in labels]); slides = np.asarray([str(x) for x in slides])
    if len(labels) != graph.shape[0] or len(slides) != len(labels):
        raise ContractError("labels/slides/graph length mismatch")
    rows, cols, weights = _undirected_edges(graph)
    all_metrics, null_rows, frag_rows = [], [], []
    for slide in sorted(set(slides)):
        indices = np.flatnonzero(slides == slide)
        local = graph[indices][:, indices]
        local_labels = labels[indices]
        er, ec, ew = _undirected_edges(local)
        observed = _within_fraction(local_labels, er, ec, ew)
        null = _permutation_values(local_labels, local, np.repeat(slide, len(indices)), permutations, seed)
        mean, sd = float(np.nanmean(null)), float(np.nanstd(null, ddof=1))
        if len(er) and (not np.isfinite(null).all() or not np.isfinite(observed)):
            raise ContractError(f"defined spatial permutation metrics are non-finite for slide {slide}")
        p = pas(local_labels, local)
        denom = 1.0 - mean
        all_metrics.append({"arm": arm, "slide": slide, "spots": len(indices), "edges": len(er), "within_domain_edge_fraction": observed, "boundary_rate": 1.0 - observed if np.isfinite(observed) else np.nan, "null_mean": mean, "null_sd": sd, "z": (observed - mean) / sd if np.isfinite(sd) and sd > 0 else np.nan, "normalized_excess_homophily": (observed - mean) / denom if denom > 0 else np.nan, "categorical_assortativity": categorical_assortativity(local_labels, local), **p})
        null_rows.extend({"arm": arm, "slide": slide, "permutation": i, "within_domain_edge_fraction": float(value), "seed": seed} for i, value in enumerate(null))
        f = fragmentation(local_labels, local); f.insert(0, "slide", slide); f.insert(0, "arm", arm); frag_rows.extend(f.to_dict("records"))
    return pd.DataFrame(all_metrics), pd.DataFrame(null_rows), pd.DataFrame(frag_rows)


def _normalize_log1p(matrix: Any) -> np.ndarray:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    totals = dense.sum(axis=1).astype(float)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ContractError("raw count library sizes must be finite and positive")
    return np.log1p(dense / totals[:, None] * 10000.0)


def select_hvgs(matrix: Any | None = None, *, n_genes: int = 2000, normalized: np.ndarray | None = None) -> list[str]:
    """Select top-variance genes without using either arm's labels."""
    values = normalized if normalized is not None else _normalize_log1p(matrix)
    variances = values.var(axis=0)
    genes = np.asarray([str(i) for i in range(values.shape[1])])
    order = np.lexsort((genes, -variances))[: min(n_genes, values.shape[1])]
    return genes[order].tolist()


def _pca_embedding(values: np.ndarray, order: np.ndarray, seed: int) -> np.ndarray:
    component_count = min(50, len(order), max(1, values.shape[0] - 1))
    return PCA(n_components=component_count, random_state=seed, svd_solver="randomized" if component_count < min(values.shape) else "auto").fit_transform(values[:, order])


def expression_silhouette(matrix: Any, labels: Iterable[Any], slides: Iterable[Any], *, n_genes: int = 2000, seed: int = DEFAULT_SEED, selected_genes: list[str] | None = None, normalized: np.ndarray | None = None, embedding: np.ndarray | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Label-free variance HVGs, fixed-seed PCA, and per-slide silhouettes."""
    values = normalized if normalized is not None else _normalize_log1p(matrix)
    labels = np.asarray([str(x) for x in labels]); slides = np.asarray([str(x) for x in slides])
    selected = selected_genes if selected_genes is not None else select_hvgs(matrix, n_genes=n_genes, normalized=values)
    order = np.asarray([int(gene) for gene in selected], dtype=int)
    embedding = embedding if embedding is not None else _pca_embedding(values, order, seed)
    rows = []
    for slide in sorted(set(slides)):
        idx = np.flatnonzero(slides == slide); y = labels[idx]
        defined = len(idx) >= 3 and 2 <= len(set(y)) < len(idx)
        score = silhouette_score(embedding[idx], y) if defined else np.nan
        if defined and not np.isfinite(score):
            raise ContractError(f"defined expression silhouette is non-finite for slide {slide}")
        rows.append({"slide": slide, "spots": len(idx), "n_domains": len(set(y)), "silhouette": float(score) if np.isfinite(score) else np.nan, "defined": bool(defined), "hvg_method": "top variance of deterministic library-normalized log1p counts", "hvg_count": len(selected), "pca_seed": seed})
    return pd.DataFrame(rows), selected


def patient_logo_signatures(matrix: Any, labels: Iterable[Any], patients: Iterable[Any], *, genes: list[str], min_heldout: int = 5, min_train: int = 20, seed: int = DEFAULT_SEED, normalized: np.ndarray | None = None) -> pd.DataFrame:
    """Descriptive patient-LOGO signatures in log-normalized expression space.

    Each signature is mean log1p library-normalized expression in the domain
    minus mean log1p expression in the same patient's non-domain background;
    it is not a log ratio of already averaged log values.
    """
    expression = normalized if normalized is not None else _normalize_log1p(matrix)
    values = expression[:, [int(g) for g in genes]]
    labels = np.asarray([str(x) for x in labels]); patients = np.asarray([str(x) for x in patients])
    domains = sorted(set(labels)); rows = []
    for patient in sorted(set(patients)):
        test = patients == patient; train = ~test
        for domain in domains:
            held_domain = test & (labels == domain); train_domain = train & (labels == domain)
            if held_domain.sum() < min_heldout or train_domain.sum() < min_train: continue
            held_bg = test & (labels != domain); train_bg = train & (labels != domain)
            if not held_bg.any() or not train_bg.any(): continue
            held_sig = values[held_domain].mean(0) - values[held_bg].mean(0)
            train_sig = values[train_domain].mean(0) - values[train_bg].mean(0)
            same = float(spearmanr(held_sig, train_sig).statistic) if len(genes) > 1 else np.nan
            others = []
            for other in domains:
                if other == domain: continue
                td = train & (labels == other)
                if td.sum() >= min_train:
                    other_sig = values[td].mean(0) - values[train & (labels != other)].mean(0)
                    others.append(float(spearmanr(held_sig, other_sig).statistic))
            best = max(others) if others else np.nan
            rows.append({"heldout_patient": patient, "domain": domain, "heldout_spots": int(held_domain.sum()), "training_spots": int(train_domain.sum()), "same_domain_spearman": same, "best_other_domain_spearman": best, "margin": same - best if np.isfinite(best) else np.nan, "selected_gene_count": len(genes), "seed": seed, "interpretation": "descriptive/exploratory cohort-derived labels"})
    return pd.DataFrame(rows)


def top_signature_markers(matrix: Any, labels: Iterable[Any], genes: list[str], gene_names: Iterable[Any], *, arm: str, top_n: int = 20, normalized: np.ndarray | None = None) -> pd.DataFrame:
    """Return descriptive top positive domain-vs-rest expression signatures."""
    expression = normalized if normalized is not None else _normalize_log1p(matrix)
    values = expression[:, [int(gene) for gene in genes]]
    labels = np.asarray([str(value) for value in labels])
    names = np.asarray([str(value) for value in gene_names])[[int(gene) for gene in genes]]
    rows = []
    for domain in sorted(set(labels)):
        in_domain = labels == domain
        background = ~in_domain
        if not background.any():
            continue
        signature = values[in_domain].mean(axis=0) - values[background].mean(axis=0)
        order = np.argsort(-signature, kind="stable")[:top_n]
        rows.extend({"arm": arm, "domain": domain, "gene": names[index], "rank": rank, "domain_vs_rest_mean_log_expression": float(signature[index]), "interpretation": "descriptive label-only signature"} for rank, index in enumerate(order, 1))
    return pd.DataFrame(rows)


# Short aliases keep the pure metric contract easy to consume from tests and notebooks.
compute_pas = pas
compute_fragmentation = fragmentation
compute_spatial_metrics = spatial_metrics
compute_shared_valid_filter = shared_valid_filter


def confounding_metrics(labels: Iterable[Any], patients: Iterable[Any]) -> dict[str, float]:
    labels = np.asarray([str(v) for v in labels]); patients = np.asarray([str(v) for v in patients])
    nmi = float(normalized_mutual_info_score(patients, labels))
    table = pd.crosstab(patients, labels).to_numpy(dtype=float)
    total = table.sum(); expected = np.outer(table.sum(1), table.sum(0)) / total
    chi2 = float(((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())
    r, k = table.shape
    cramers = float(np.sqrt((chi2 / total) / max(1, min(k - 1, r - 1)))) if total else np.nan
    return {"patient_domain_nmi": nmi, "patient_domain_cramers_v": cramers, "interpretation": "descriptive; lower is not automatically biologically better"}


def validate_provenance(uns: Mapping[str, Any], *, expected_hashes: Mapping[str, str] | None = None) -> None:
    """Require calibrated NOVAE provenance, including reference=all and L0-L8."""
    payload: Mapping[str, Any] = uns.get("novae_pilot_provenance", uns) if isinstance(uns, Mapping) else {}
    required = {"analysis_scope": "exploratory", "reference": "all", "dataset_id": "skin_visium_ssc_paired_cpu_calibrated", "coordinate_strategy": "visium_explicit_scale", "domain_key": DOMAIN_KEY, "neighborhood_valid_key": VALID_KEY, "primary_resolution": 1.0, "accelerator": "cpu", "device": "cpu", "workers": 0, "seed": 42}
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ContractError(f"NOVAE provenance mismatch for {key}")
    input_hash = payload.get("input_sha256")
    checkpoint_hash = payload.get("checkpoint_sha256")
    if input_hash != EXPECTED_NOVAE_INPUT_SHA256 or checkpoint_hash != EXPECTED_NOVAE_CHECKPOINT_SHA256:
        raise ContractError("NOVAE provenance input/checkpoint hash does not match the calibrated contract")
    policy = payload.get("deterministic_policy", {})
    requested = policy.get("requested") if isinstance(policy, Mapping) else None
    effective = policy.get("effective") if isinstance(policy, Mapping) else None
    boolean_type = (bool, np.bool_)
    if not isinstance(requested, boolean_type) or not isinstance(effective, boolean_type) or not bool(requested) or not bool(effective):
        raise ContractError("NOVAE provenance deterministic policy is not effective")
    if expected_hashes:
        hashes = payload.get("hashes", payload.get("input_hashes", {}))
        for key, expected in expected_hashes.items():
            if hashes.get(key) != expected:
                raise ContractError(f"NOVAE provenance hash mismatch for {key}")


def _load_anndata(path: Path) -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anndata is required on the SLURM environment") from exc
    return ad.read_h5ad(path)


def run_validation(novae_h5ad: str | Path, post_nmf_obs: str | Path, output_dir: str | Path, *, permutations: int = DEFAULT_PERMUTATIONS, seed: int = DEFAULT_SEED, expected_valid: int | None = EXPECTED_VALID, expected_invalid: int | None = EXPECTED_INVALID, expected_h5ad_sha256: str = EXPECTED_CALIBRATED_H5AD_SHA256, expected_post_sha256: str = EXPECTED_POST_NMF_OBS_SHA256) -> Path:
    """Execute validation and atomically publish a complete report directory.

    The expected hash parameters are dependency-injection seams for synthetic
    tests; they are intentionally not exposed as CLI options.
    """
    if permutations < 2:
        raise ContractError("permutations must be at least 2")
    h5ad, post, output = Path(novae_h5ad), Path(post_nmf_obs), Path(output_dir)
    if output.exists(): raise ContractError(f"refusing existing output: {output}")
    if not h5ad.is_file() or not post.is_file(): raise ContractError("required input does not exist")
    h5ad_sha256, post_sha256 = sha256(h5ad), sha256(post)
    if h5ad_sha256 != expected_h5ad_sha256:
        raise ContractError("calibrated NOVAE H5AD SHA256 does not match the frozen input")
    if post_sha256 != expected_post_sha256:
        raise ContractError("post_nmf_obs.csv SHA256 does not match the frozen input")
    adata = _load_anndata(h5ad)
    ids = _text(adata.obs["unique_cell_id"] if "unique_cell_id" in adata.obs else adata.obs_names, "H5AD unique_cell_id")
    if not pd.Index(ids).is_unique or not pd.Index(adata.obs_names.astype(str)).is_unique: raise ContractError("H5AD IDs must be unique")
    if "unique_cell_id" in adata.obs and not np.array_equal(ids, adata.obs_names.astype(str)): raise ContractError("obs_names and unique_cell_id differ")
    if DOMAIN_KEY not in adata.obs or VALID_KEY not in adata.obs: raise ContractError("calibrated NOVAE domain/validity columns are missing")
    valid_all = _bool_values(adata.obs[VALID_KEY], VALID_KEY)
    raw_domains = adata.obs[DOMAIN_KEY].tolist()
    novae_labels_all = np.asarray([str(value).strip() if not pd.isna(value) else "" for value in raw_domains], dtype=object)
    if set(novae_labels_all[valid_all]) != EXPECTED_DOMAINS:
        raise ContractError("valid NOVAE domain vocabulary must contain exactly L0-L8")
    if (novae_labels_all[~valid_all] != "").any():
        raise ContractError("invalid NOVAE rows must have no domain assignment")
    validate_provenance(adata.uns)
    coords = _finite_coords(adata); counts = _raw_counts(adata)
    slide_key = _pick(adata.obs.columns, ("sample_id", "slide", "library_id"), "H5AD slide/sample")
    patient_key = _pick(adata.obs.columns, ("patient", "Patient", "subject"), "H5AD patient")
    disease_key = _pick(adata.obs.columns, ("Disease_State", "disease_state", "Disease/Health State", "Disease.Health.State"), "H5AD disease")
    slides = _text(adata.obs[slide_key], slide_key); patients = _text(adata.obs[patient_key], patient_key); disease = _text(adata.obs[disease_key], disease_key)
    if GRAPH_KEY not in adata.obsp:
        raise ContractError(f"obsp[{GRAPH_KEY}] is missing")
    graph = validate_graph(adata.obsp[GRAPH_KEY], slides, len(ids))
    post_df = pd.read_csv(post)
    id_col = _pick(post_df.columns, ("unique_cell_id", "unique_cellid", "cell_id", "cell_ID"), "post_nmf_obs unique_cell_id")
    factor_col = _pick(post_df.columns, ("NMF_factor", "nmf_factor", "dominant_nmf_factor"), "NMF_factor")
    post_ids = _text(post_df[id_col], "post_nmf_obs unique_cell_id")
    if len(post_ids) != len(set(post_ids)): raise ContractError("post_nmf_obs unique_cell_id is not unique")
    by_id = post_df.assign(_id=post_ids).set_index("_id")
    if set(by_id.index) != set(ids): raise ContractError("post_nmf_obs IDs do not exactly match H5AD")
    nmf = _factor_values(by_id.loc[ids, factor_col], "NMF_factor")
    if set(nmf) != EXPECTED_NMF_FACTORS: raise ContractError("NMF factor vocabulary must contain exactly 0-8")
    post_patient_key = _pick(by_id.columns, ("patient", "Patient", "subject"), "post_nmf_obs patient")
    post_disease_key = _pick(by_id.columns, ("Disease_State", "disease_state", "Disease/Health State", "Disease.Health.State"), "post_nmf_obs disease")
    post_patients = _text(by_id.loc[ids, post_patient_key], f"post_nmf_obs {post_patient_key}")
    post_disease = _text(by_id.loc[ids, post_disease_key], f"post_nmf_obs {post_disease_key}")
    if not np.array_equal(post_patients, patients):
        raise ContractError("post_nmf_obs patient metadata disagrees with H5AD")
    if not np.array_equal(post_disease, disease):
        raise ContractError("post_nmf_obs disease metadata disagrees with H5AD")
    # A slide/sample comparison is required only where both files carry the
    # same documented key; field_of_view is not assumed to equal sample_id.
    for shared_key in ("sample_id", "slide"):
        if shared_key in by_id.columns and shared_key in adata.obs.columns:
            post_values = _text(by_id.loc[ids, shared_key], f"post_nmf_obs {shared_key}")
            h5_values = _text(adata.obs[shared_key], f"H5AD {shared_key}")
            if not np.array_equal(post_values, h5_values):
                raise ContractError(f"post_nmf_obs {shared_key} metadata disagrees with H5AD")
    mask, coverage = shared_valid_filter(valid_all, ids, ids, expected_valid=expected_valid, expected_invalid=expected_invalid)
    order = np.flatnonzero(mask); shared_graph = graph[order][:, order]; shared_slides = slides[order]; shared_patients = patients[order]
    output.parent.mkdir(parents=True, exist_ok=True)
    arms = {"nmf": nmf[order], "novae": novae_labels_all[order]}
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        spatial_frames = []; null_frames = []; frag_frames = []
        for arm, labels in arms.items():
            sm, pn, fm = spatial_metrics(labels, shared_graph, shared_slides, permutations=permutations, seed=seed, arm=arm)
            spatial_frames.append(sm); null_frames.append(pn); frag_frames.append(fm)
        spatial = pd.concat(spatial_frames, ignore_index=True); nulls = pd.concat(null_frames, ignore_index=True); frag = pd.concat(frag_frames, ignore_index=True)
        expression_frames = []; signature_frames = []; marker_frames = []
        shared_counts = counts[order]
        # Normalize and select HVGs once: both arms receive exactly the same
        # expression space, avoiding two full dense copies and two PCA/HVG fits.
        shared_expression = _normalize_log1p(shared_counts)
        selected_genes = select_hvgs(normalized=shared_expression)
        selected_indices = np.asarray([int(gene) for gene in selected_genes], dtype=int)
        shared_embedding = _pca_embedding(shared_expression, selected_indices, seed)
        for arm, labels in arms.items():
            sil, _ = expression_silhouette(shared_counts, labels, shared_slides, seed=seed, selected_genes=selected_genes, normalized=shared_expression, embedding=shared_embedding)
            sil.insert(0, "arm", arm); expression_frames.append(sil)
            signature_frames.append(patient_logo_signatures(shared_counts, labels, shared_patients, genes=selected_genes, seed=seed, normalized=shared_expression).assign(arm=arm))
            marker_frames.append(top_signature_markers(shared_counts, labels, selected_genes, adata.var_names, arm=arm, top_n=20, normalized=shared_expression))
        coverage_table = pd.DataFrame({"unique_cell_id": ids[order], "slide": shared_slides, "patient": shared_patients, "x": coords[order, 0], "y": coords[order, 1], "novae_valid": True, "nmf_factor": nmf[order], "novae_domain": novae_labels_all[order]})
        prevalence = coverage_table.melt(id_vars=["unique_cell_id", "slide", "patient"], value_vars=["nmf_factor", "novae_domain"], var_name="arm", value_name="domain").groupby(["arm", "slide", "domain"], as_index=False).size().rename(columns={"size": "spots"})
        arm_summaries = {}
        slide_summary = spatial.merge(pd.concat(expression_frames, ignore_index=True), on=["arm", "slide"], how="left")
        summary_metrics = ("within_domain_edge_fraction", "boundary_rate", "null_mean", "null_sd", "z", "normalized_excess_homophily", "categorical_assortativity", "pas", "silhouette")
        def _mean_defined(values: pd.Series) -> float:
            numeric = values.to_numpy(dtype=float)
            return float(np.mean(numeric[np.isfinite(numeric)])) if np.isfinite(numeric).any() else np.nan

        for arm in arms:
            arm_rows = slide_summary.loc[slide_summary.arm == arm].copy()
            edge_weight = arm_rows["edges"].to_numpy(dtype=float)
            values = arm_rows["within_domain_edge_fraction"].to_numpy(dtype=float)
            arm_summaries[arm] = {f"unweighted_slide_mean_{metric}": _mean_defined(arm_rows[metric]) for metric in summary_metrics}
            weighted_within = float(np.average(values, weights=edge_weight)) if edge_weight.sum() else np.nan
            arm_summaries[arm]["weighted_slide_mean_within_domain_edge_fraction_descriptive"] = weighted_within
            arm_summaries[arm]["weighted_slide_mean_within_edge_fraction"] = weighted_within
            arm_summaries[arm]["slide_count"] = int(len(arm_rows))
        summary = {"contract": {"K": 9, "valid_rows": len(order), "invalid_rows": int((~valid_all).sum()), "graph": GRAPH_KEY, "complete_case": True, "no_imputation": True, "seed": seed, "permutations": permutations}, "arms": arm_summaries, "confounding": {arm: confounding_metrics(labels, shared_patients) for arm, labels in arms.items()}, "coverage": coverage}
        confounds = pd.DataFrame([{"arm": arm, **confounding_metrics(labels, shared_patients)} for arm, labels in arms.items()])
        summary_csv = pd.DataFrame([{"arm": arm, **values} for arm, values in arm_summaries.items()])
        graph_contract = {"graph_key": GRAPH_KEY, "topology": "identical induced undirected graph for both arms", "nodes": len(order), "undirected_edges": len(_undirected_edges(shared_graph)[0]), "cross_slide_edges": 0, "coordinates": "identical shared x/y", "expression": "identical shared raw-count matrix", "zero_degree_spots": int((np.diff(shared_graph.indptr) == 0).sum())}
        _atomic_table(coverage_table, stage / "shared_observation_graph_contract.parquet"); _atomic_json(graph_contract, stage / "graph_contract.json"); _atomic_table(spatial, stage / "spatial_metrics_per_slide.parquet"); _atomic_table(slide_summary, stage / "unweighted_per_slide_summary.parquet"); _atomic_table(frag, stage / "fragmentation_per_slide_domain.parquet"); _atomic_table(nulls, stage / "permutation_nulls.parquet"); _atomic_table(pd.concat(expression_frames, ignore_index=True), stage / "expression_silhouette.parquet"); _atomic_table(pd.concat(signature_frames, ignore_index=True), stage / "patient_logo_signatures.parquet"); _atomic_table(prevalence, stage / "coverage_prevalence.parquet"); _atomic_table(confounds, stage / "confounding_metrics.parquet"); _atomic_json(summary, stage / "method_summary.json"); _atomic_table(summary_csv, stage / "method_summary.csv")
        _atomic_json({"cell_type_coherence": {"status": "blocked", "reason": "NMF derives from cell2location/dominant type; circular without independent annotations"}, "pathway_histology": {"status": "blocked", "reason": "not available"}, "seed_stability": {"status": "blocked", "reason": "no comparable NMF reruns"}}, stage / "blocked_status.json")
        var_names = np.asarray([str(value) for value in adata.var_names])
        marker_table = pd.concat(marker_frames, ignore_index=True)
        _atomic_table(marker_table, stage / "top_marker_signature_genes.csv")
        hvg_table = pd.DataFrame({"rank": np.arange(1, len(selected_genes) + 1), "gene": [var_names[int(gene)] for gene in selected_genes], "feature_index": [int(gene) for gene in selected_genes], "selection": "label-free top variance of shared log-normalized expression"})
        _atomic_table(hvg_table, stage / "selected_hvgs.csv")
        manifest = {"contract": "novae_spatial_biological_validation", "inputs": {"novae_h5ad": {"path": str(h5ad), "sha256": h5ad_sha256}, "post_nmf_obs": {"path": str(post), "sha256": post_sha256}}, "outputs": {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}, "caveats": ["descriptive/exploratory; cohort-derived labels", "disease labels never used", "no naive FOV p-values"], "code_sha256": sha256(Path(__file__))}
        _atomic_json(manifest, stage / "manifest.json")
        output.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novae-h5ad", required=True, type=Path)
    parser.add_argument("--post-nmf-obs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_validation(args.novae_h5ad, args.post_nmf_obs, args.output_dir, permutations=args.permutations, seed=args.seed)
    except (ContractError, RuntimeError, OSError, ValueError) as exc:
        print(f"validation blocked: {exc}", file=os.sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
