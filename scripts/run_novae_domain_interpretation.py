#!/usr/bin/env python3
"""Read-only exploratory NMF/NOVAE skin-domain interpretation.

This stage consumes the frozen calibrated NOVAE H5AD, its exact paired
``post_nmf_obs`` table, and the completed spatial-validation report.  It is
intentionally descriptive: no cell type is assigned and disease is never used
for marker/program/pathway selection.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

N_DOMAINS = 9
N_VALID = 13372
N_INVALID = 45
EXPECTED_CALIBRATED_H5AD_SHA256 = "40b60eba32c0716637eae98a28c852c11c53dfe4168570ad3230cf0d43219ffd"
EXPECTED_POST_NMF_OBS_SHA256 = "a79a4e5949752110593f45eccd1ff34786b7d39cea3ce144b635444638bb354b"
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAPS = 10_000
MIN_DOMAIN_SPOTS = 5
MIN_BACKGROUND_SPOTS = 20
PROGRAM_COVERAGE_GATE = 0.5
PROGRAM_TOP_N = 5
PROGRAM_RESOURCE = "presets/novae_skin_marker_programs.csv"
HALLMARK_RESOURCE = "presets/h.all.v2025.1.Hs.symbols.gmt"
RESOURCE_MANIFEST = "presets/novae_skin_resources.json"
HALLMARK_SIDECAR = "presets/h.all.v2025.1.Hs.symbols.gmt.sha256"
HALLMARK_SHA256 = "f22066af72e215ccb7b89d88e492c07e1eef17534c2ca7b0f9902cfecbbdd8e9"
PROGRAM_SHA256 = "402af0d62aca43a5b90d3ac4b0443b54b3dc6c6b8c7e538a72dd9871a90317f2"
HALLMARK_SET_COUNT = 50
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resource_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.exists() else REPO_ROOT / path

class ContractError(ValueError):
    """Fail-closed interpretation contract error."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def _atomic_json(value: Any, path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, default=_json_default)
            f.write("\n")
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


def _text(values: Iterable[Any], label: str) -> np.ndarray:
    out = np.asarray([str(x).strip() for x in values], dtype=object)
    if (out == "").any() or np.isin(np.char.lower(out.astype(str)), ["nan", "none", "null"]).any():
        raise ContractError(f"{label} contains missing/blank values")
    return out


def _counts_array(counts: Any) -> np.ndarray:
    x = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x, np.rint(x)):
        raise ContractError("raw counts must be finite nonnegative integers")
    return x.astype(float, copy=False)


def eligible_detected_genes(counts: Any, *, detection_fraction: float = 0.01) -> np.ndarray:
    """Label-free genes detected in at least 1% of shared spots."""
    x = _counts_array(counts)
    if not 0 < detection_fraction <= 1:
        raise ContractError("detection_fraction must be in (0, 1]")
    minimum = max(1, int(np.ceil(detection_fraction * x.shape[0])))
    return np.flatnonzero((x > 0).sum(axis=0) >= minimum)


def library_cpm(counts: Any) -> np.ndarray:
    x = _counts_array(counts)
    totals = x.sum(axis=1)
    if (totals <= 0).any():
        raise ContractError("raw count library sizes must be positive")
    return x / totals[:, None] * 1_000_000.0


def patient_domain_signatures(
    counts: Any, labels: Iterable[Any], patients: Iterable[Any], genes: Sequence[int],
    *, arm: str = "arm", min_domain: int = MIN_DOMAIN_SPOTS,
    min_background: int = MIN_BACKGROUND_SPOTS, gene_names: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Pseudobulk CPM patient-domain minus same-patient-background signatures."""
    x = _counts_array(counts)
    labels = _text(labels, "domain labels"); patients = _text(patients, "patients")
    if len(labels) != len(x) or len(patients) != len(x):
        raise ContractError("counts/labels/patients length mismatch")
    genes = np.asarray(genes, dtype=int)
    if (genes < 0).any() or (genes >= x.shape[1]).any():
        raise ContractError("signature gene index out of range")
    names = np.asarray(gene_names if gene_names is not None else [str(i) for i in range(x.shape[1])], dtype=object)
    cpm = library_cpm(x)
    rows: list[dict[str, Any]] = []
    for patient in sorted(set(patients)):
        patient_mask = patients == patient
        for domain in sorted(set(labels)):
            dm = patient_mask & (labels == domain); bg = patient_mask & (labels != domain)
            if dm.sum() < min_domain or bg.sum() < min_background:
                continue
            # Pseudobulk means total counts, then CPM; no imputation.
            dcounts = x[dm][:, genes].sum(axis=0); bcounts = x[bg][:, genes].sum(axis=0)
            # CPM denominators are total raw counts over every gene, not just
            # the predeclared eligible/signature genes.
            d_total = x[dm].sum(); b_total = x[bg].sum()
            d_cpm = dcounts / d_total * 1_000_000 if d_total else np.zeros(len(genes))
            b_cpm = bcounts / b_total * 1_000_000 if b_total else np.zeros(len(genes))
            logfc = np.log2(d_cpm + 1.0) - np.log2(b_cpm + 1.0)
            rows.extend({"arm": arm, "patient": patient, "domain": domain, "gene": str(names[g]),
                         "gene_index": int(g), "log2_cpm_domain_minus_background": float(v),
                         "domain_spots": int(dm.sum()), "background_spots": int(bg.sum())}
                        for g, v in zip(genes, logfc, strict=True))
    return pd.DataFrame(rows)


def aggregate_patient_signatures(signatures: pd.DataFrame, *, min_patients: int = 3) -> pd.DataFrame:
    """Aggregate patient signatures and deterministic rank-stability summaries."""
    required = {"arm", "domain", "patient", "gene", "log2_cpm_domain_minus_background"}
    if not required.issubset(signatures.columns):
        raise ContractError(f"signature table missing {sorted(required - set(signatures.columns))}")
    ranked = signatures.copy()
    ranked["_within_patient_rank"] = ranked.groupby(["arm", "patient", "domain"], sort=False)["log2_cpm_domain_minus_background"].rank(method="average", ascending=False)
    rank_width = ranked.groupby(["arm", "patient", "domain"], sort=False).gene.transform("nunique").clip(lower=1)
    ranked["_rank_fraction"] = (ranked["_within_patient_rank"] - 1) / (rank_width - 1).replace(0, 1)
    rows = []
    for (arm, domain, gene), group in ranked.groupby(["arm", "domain", "gene"], sort=True):
        by_patient = group.groupby("patient", sort=True).agg(value=("log2_cpm_domain_minus_background", "mean"), rank=("_rank_fraction", "mean"))
        values = by_patient["value"].to_numpy(float); ranks = by_patient["rank"].to_numpy(float)
        if len(values) < min_patients:
            continue
        med = float(np.median(values)); q25, q75 = np.quantile(values, [0.25, 0.75])
        rank_iqr = float(np.subtract(*np.quantile(ranks, [0.75, 0.25]))) if len(ranks) else np.nan
        stability = float(1.0 - rank_iqr) if len(ranks) else np.nan
        rows.append({"arm": arm, "domain": domain, "gene": gene, "patient_count": len(values),
                     "median_logFC": med, "q25_logFC": float(q25), "q75_logFC": float(q75),
                     "IQR_logFC": float(q75 - q25), "fraction_positive": float(np.mean(values > 0)),
                     "median_within_patient_rank_fraction": float(np.median(ranks)), "rank_IQR": rank_iqr,
                     "rank_stability": stability})
    return pd.DataFrame(rows)


def top_positive_markers(aggregated: pd.DataFrame, *, top_n: int = 100) -> pd.DataFrame:
    """Apply the predeclared positive-marker gate and deterministic ranking."""
    if aggregated.empty:
        return aggregated.copy()
    selected = aggregated[(aggregated.median_logFC > 0) & (aggregated.fraction_positive >= 0.6)].copy()
    selected = selected.sort_values(["arm", "domain", "median_logFC", "fraction_positive", "gene"],
                                    ascending=[True, True, False, False, True], kind="mergesort")
    selected["rank"] = selected.groupby(["arm", "domain"], sort=False).cumcount() + 1
    return selected[selected["rank"] <= top_n].reset_index(drop=True)


def read_programs(path: str | Path = PROGRAM_RESOURCE) -> pd.DataFrame:
    frame = pd.read_csv(_resource_path(path))
    required = {"category", "program", "source_id", "genes"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ContractError("marker program resource has invalid columns")
    frame = frame.copy(); frame["genes"] = frame.genes.map(lambda x: tuple(sorted({g.strip() for g in str(x).split(";") if g.strip()})))
    if (frame.genes.map(len) < 2).any() or frame.program.duplicated().any():
        raise ContractError("programs must be unique multi-gene programs")
    return frame


def score_programs(aggregated: pd.DataFrame, programs: pd.DataFrame | str | Path = PROGRAM_RESOURCE) -> pd.DataFrame:
    """Score fixed programs from patient-aware gene effects, reporting coverage."""
    p = read_programs(programs) if not isinstance(programs, pd.DataFrame) else programs
    rows = []
    for (arm, domain), group in aggregated.groupby(["arm", "domain"], sort=True):
        effects = dict(zip(group.gene.astype(str), group.median_logFC.astype(float), strict=True))
        for _, program in p.iterrows():
            genes = list(program.genes); observed = [effects[g] for g in genes if g in effects]
            rows.append({"arm": arm, "domain": domain, "category": program.category, "program": program.program,
                         "source_id": program.source_id, "gene_count": len(genes), "covered_genes": len(observed),
                         "coverage": len(observed) / len(genes), "score": float(np.mean(observed)) if observed else np.nan,
                         "positive_covered_genes": int(sum(v > 0 for v in observed))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["arm", "domain", "score", "coverage", "program"], ascending=[True, True, False, False, True], na_position="last", kind="mergesort")
        out["candidate_rank"] = out.groupby(["arm", "domain"], sort=False).cumcount() + 1
    return out


def top_program_candidates(scores: pd.DataFrame, *, coverage_gate: float = PROGRAM_COVERAGE_GATE, top_n: int = PROGRAM_TOP_N) -> pd.DataFrame:
    """Select only positive fixed-program candidates under the predeclared gate."""
    required = {"arm", "domain", "coverage", "score"}
    if not required.issubset(scores.columns): raise ContractError("program score table missing candidate columns")
    selected = scores[(scores.coverage >= coverage_gate) & scores.score.notna() & (scores.score > 0)].copy()
    if selected.empty:
        selected["candidate_rank"] = pd.Series(dtype=int)
        return selected
    selected = selected.sort_values(["arm", "domain", "score", "coverage", "program"], ascending=[True, True, False, False, True], kind="mergesort")
    selected["candidate_rank"] = selected.groupby(["arm", "domain"], sort=False).cumcount() + 1
    return selected[selected.candidate_rank <= top_n].reset_index(drop=True)


def parse_gmt(path: str | Path) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    with _resource_path(path).open(encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                sets[fields[0]] = {g.strip() for g in fields[2:] if g.strip()}
    if not sets:
        raise ContractError("GMT contains no gene sets")
    return sets


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float); out = np.full(len(p), np.nan)
    finite = np.isfinite(p); idx = np.flatnonzero(finite)
    if len(idx):
        order = idx[np.argsort(p[idx], kind="stable")]; ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return out


def hallmark_ora(top_markers: pd.DataFrame, eligible_genes: Iterable[Any], gmt: Mapping[str, set[str]] | str | Path = HALLMARK_RESOURCE) -> pd.DataFrame:
    """Exact hypergeometric ORA and BH within each arm/domain."""
    sets = parse_gmt(gmt) if not isinstance(gmt, Mapping) else gmt
    background = {str(g) for g in eligible_genes}; rows = []
    for (arm, domain), group in top_markers.groupby(["arm", "domain"], sort=True):
        selected = set(group.gene.astype(str)) & background
        for name in sorted(sets):
            members = sets[name] & background; overlap = selected & members; M = len(background); n = len(members); N = len(selected); k = len(overlap)
            p = float(hypergeom.sf(k - 1, M, n, N)) if M and n and N else 1.0
            rows.append({"arm": arm, "domain": domain, "gene_set": name, "overlap_count": k,
                         "overlap_genes": ";".join(sorted(overlap)), "selected_count": N, "background_count": M,
                         "gene_set_background_count": n, "p_value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = out.groupby(["arm", "domain"], sort=False).p_value.transform(lambda x: bh_adjust(x.to_numpy()))
    return out


def signature_similarity(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """9x9 Spearman similarity on common genes; no disease input."""
    common = sorted(set(left.gene) & set(right.gene))
    if not common: raise ContractError("NMF/NOVAE have no common eligible genes")
    arms = {str(x) for x in left.arm} | {str(x) for x in right.arm}
    if len(arms) != 2: raise ContractError("similarity requires two arms")
    a, b = sorted(arms); rows = []
    for da in sorted(left[left.arm == a].domain.unique()):
        av = left[(left.arm == a) & left.domain.eq(da)].set_index("gene").reindex(common).median_logFC
        for db in sorted(right[right.arm == b].domain.unique()):
            bv = right[(right.arm == b) & right.domain.eq(db)].set_index("gene").reindex(common).median_logFC
            keep = np.isfinite(av.to_numpy(float)) & np.isfinite(bv.to_numpy(float)); av2, bv2 = av.to_numpy(float)[keep], bv.to_numpy(float)[keep]
            rows.append({"left_arm": a, "left_domain": da, "right_arm": b, "right_domain": db,
                         "common_gene_count": int(keep.sum()), "spearman": float(spearmanr(av2, bv2).statistic) if len(av2) > 1 else np.nan})
    return pd.DataFrame(rows)


def hungarian_matching(similarity: pd.DataFrame) -> pd.DataFrame:
    """Deterministic maximum assignment; row/column orientation is explicit."""
    if similarity.empty: return similarity.copy()
    rows = sorted(similarity.left_domain.unique()); cols = sorted(similarity.right_domain.unique())
    matrix = np.full((len(rows), len(cols)), -1.0)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            value = similarity[(similarity.left_domain == r) & (similarity.right_domain == c)].spearman
            matrix[i, j] = float(value.iloc[0]) if len(value) and np.isfinite(value.iloc[0]) else -1.0
    # Tiny lexicographic perturbation resolves equal scores without changing ordering.
    cost = -matrix + np.arange(len(rows))[:, None] * 1e-12 + np.arange(len(cols))[None, :] * 1e-15
    rr, cc = linear_sum_assignment(cost)
    return pd.DataFrame({"nmf_domain": [rows[i] for i in rr], "novae_domain": [cols[j] for j in cc],
                         "spearman": [matrix[i, j] for i, j in zip(rr, cc, strict=True)], "matching": "Hungarian maximum; descriptive only"})


def best_signature_matches(similarity: pd.DataFrame) -> pd.DataFrame:
    """Return non-exclusive deterministic best matches in both directions.

    This intentionally differs from the one-to-one Hungarian assignment: a
    domain may be the best match for multiple domains in this descriptive view.
    """
    required = {"left_domain", "right_domain", "spearman"}
    if not required.issubset(similarity.columns): raise ContractError("similarity table missing match columns")
    rows = []
    for direction, query_col, target_col in (("novae_to_nmf", "right_domain", "left_domain"), ("nmf_to_novae", "left_domain", "right_domain")):
        for query in sorted(similarity[query_col].astype(str).unique()):
            candidates = similarity[similarity[query_col].astype(str) == query].copy()
            candidates["_score"] = candidates.spearman.fillna(-np.inf)
            candidates = candidates.sort_values(["_score", target_col], ascending=[False, True], kind="mergesort")
            if candidates.empty: continue
            best_value = candidates.iloc[0]["spearman"]
            best = float(best_value) if np.isfinite(best_value) else np.nan
            ties = candidates[np.isclose(candidates["_score"].to_numpy(float), candidates.iloc[0]["_score"], rtol=0, atol=1e-12)]
            rows.append({"direction": direction, "query_domain": query, "best_domain": str(candidates.iloc[0][target_col]), "spearman": best, "tie_count": int(len(ties)), "matching": "non-exclusive best from full matrix; distinct from Hungarian"})
    return pd.DataFrame(rows)


def spot_agreement(nmf: Iterable[Any], novae: Iterable[Any], patients: Iterable[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contingency/Jaccard and global/per-patient ARI/NMI agreement."""
    a = _text(nmf, "NMF labels"); b = _text(novae, "NOVAE labels"); p = _text(patients, "patients")
    if not (len(a) == len(b) == len(p)): raise ContractError("agreement lengths differ")
    contingency = pd.crosstab(pd.Series(a, name="nmf_domain"), pd.Series(b, name="novae_domain"), dropna=False).stack().reset_index(name="spots")
    rows = []
    groups = [("global", np.ones(len(a), dtype=bool))] + [(str(x), p == x) for x in sorted(set(p))]
    for group, mask in groups:
        rows.append({"group": group, "spots": int(mask.sum()), "ARI": float(adjusted_rand_score(a[mask], b[mask])), "NMI": float(normalized_mutual_info_score(a[mask], b[mask]))})
    return contingency, pd.DataFrame(rows)


def matched_overlap(nmf: Iterable[Any], novae: Iterable[Any], patients: Iterable[Any], matching: pd.DataFrame) -> pd.DataFrame:
    a = _text(nmf, "NMF labels"); b = _text(novae, "NOVAE labels"); p = _text(patients, "patients")
    rows = []
    for r in matching.itertuples(index=False):
        masks = [("global", np.ones(len(a), bool))] + [(str(x), p == x) for x in sorted(set(p))]
        for group, subset in masks:
            x = subset & (a == str(r.nmf_domain)); y = subset & (b == str(r.novae_domain)); inter = int((x & y).sum()); union = int((x | y).sum())
            rows.append({"group": group, "nmf_domain": r.nmf_domain, "novae_domain": r.novae_domain, "nmf_spots": int(x.sum()), "novae_spots": int(y.sum()), "overlap_spots": inter, "jaccard": inter / union if union else np.nan})
    return pd.DataFrame(rows)


def complete_patient_prevalence(coverage: pd.DataFrame, *, label_columns: Mapping[str, str] | None = None, domains: Mapping[str, Sequence[str]] | None = None) -> pd.DataFrame:
    """Return complete arm×patient×domain prevalence, retaining structural zeros."""
    label_columns = label_columns or {"nmf": "nmf_factor", "novae": "novae_domain"}
    domains = domains or {"nmf": [str(i) for i in range(N_DOMAINS)], "novae": [f"L{i}" for i in range(N_DOMAINS)]}
    patient_frame = coverage[["patient", "disease"]].drop_duplicates().sort_values("patient")
    if patient_frame.patient.nunique() != len(patient_frame): raise ContractError("patient has multiple disease metadata rows")
    frames = []
    for arm, column in label_columns.items():
        grid = patient_frame.assign(_key=1).merge(pd.DataFrame({"domain": list(domains[arm]), "_key": 1}), on="_key").drop(columns="_key")
        observed = coverage.groupby(["patient", "disease", column], as_index=False).size().rename(columns={column: "domain", "size": "spots"})
        grid = grid.merge(observed, on=["patient", "disease", "domain"], how="left").fillna({"spots": 0})
        grid["arm"] = arm; grid["spots"] = grid.spots.astype(int); grid["total_patient_spots"] = coverage.groupby("patient").size().reindex(grid.patient).to_numpy(); grid["proportion"] = grid.spots / grid.total_patient_spots
        frames.append(grid)
    return pd.concat(frames, ignore_index=True)


def exact_prevalence_test(values: pd.DataFrame, *, disease_col: str = "disease", patient_col: str = "patient", value_col: str = "proportion", healthy_label: str = "healthy", ssc_label: str = "systemic_sclerosis") -> pd.DataFrame:
    """Enumerate every fixed 4-healthy/10-SSc assignment for each domain."""
    patient = values.groupby(["domain", patient_col, disease_col], as_index=False)[value_col].mean()
    patients = sorted(patient[patient_col].astype(str).unique()); patient_labels = patient.groupby(patient_col)[disease_col].nunique()
    n_healthy = int(patient.drop_duplicates(patient_col)[disease_col].eq(healthy_label).sum())
    if len(patients) != 14 or not patient_labels.eq(1).all() or n_healthy != 4: raise ContractError("exact prevalence test requires 14 patients and 4 healthy patients")
    if set(patient.drop_duplicates(patient_col)[disease_col]) != {healthy_label, ssc_label}: raise ContractError("disease labels must be healthy and systemic_sclerosis")
    domains = sorted(values.domain.astype(str).unique()); rows = []
    for domain in domains:
        one = patient[patient.domain.astype(str) == domain].set_index(patient_col)
        if len(one) != 14: raise ContractError(f"domain {domain} does not have one value per patient")
        x = one[value_col].astype(float).reindex(patients)
        observed_means = one.groupby(disease_col)[value_col].mean()
        observed = float(observed_means.get(ssc_label, np.nan) - observed_means.get(healthy_label, np.nan))
        effects = []
        for healthy_idx in itertools.combinations(range(14), n_healthy):
            mask = np.zeros(14, bool); mask[list(healthy_idx)] = True
            effects.append(float(x.iloc[~mask].mean() - x.iloc[mask].mean()))
        effects = np.asarray(effects); tolerance = 1e-12 * max(1.0, abs(observed), float(np.max(np.abs(effects))))
        rows.append({"domain": domain, "observed_effect_ssc_minus_healthy": observed, "permutations": len(effects), "p_two_sided": float(np.mean(np.abs(effects) >= abs(observed) - tolerance)), "healthy_assignments": len(effects)})
    out = pd.DataFrame(rows); out["q_value"] = bh_adjust(out.p_two_sided.to_numpy()) if not out.empty else []
    return out


def prevalence_bootstrap(values: pd.DataFrame, *, reps: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED, disease_col: str = "disease", patient_col: str = "patient", value_col: str = "proportion", healthy_label: str = "healthy", ssc_label: str = "systemic_sclerosis") -> pd.DataFrame:
    if reps < 1: raise ContractError("bootstrap reps must be positive")
    rng = np.random.default_rng(seed); rows = []
    for domain in sorted(values.domain.astype(str).unique()):
        by = values[values.domain.astype(str) == domain].groupby([patient_col, disease_col], as_index=False)[value_col].mean()
        groups = {d: g[value_col].to_numpy(float) for d, g in by.groupby(disease_col)}
        if healthy_label not in groups or ssc_label not in groups: raise ContractError("bootstrap requires healthy and systemic_sclerosis patient groups")
        if len(groups[healthy_label]) != 4 or len(groups[ssc_label]) != 10: raise ContractError("bootstrap requires 4 healthy and 10 systemic_sclerosis patients")
        diffs = np.empty(reps)
        for i in range(reps): diffs[i] = rng.choice(groups[ssc_label], len(groups[ssc_label]), replace=True).mean() - rng.choice(groups[healthy_label], len(groups[healthy_label]), replace=True).mean()
        rows.append({"domain": domain, "bootstrap_reps": reps, "seed": seed, "effect": float(groups[ssc_label].mean() - groups[healthy_label].mean()), "ci95_low": float(np.quantile(diffs, .025)), "ci95_high": float(np.quantile(diffs, .975)), "interpretation": "descriptive stratified patient bootstrap"})
    return pd.DataFrame(rows)


def stability_summary(path: str | Path, *, min_patients: int = 1) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"arm", "domain", "same_domain_spearman", "best_other_domain_spearman", "margin"}
    if not required.issubset(frame.columns): raise ContractError("patient_logo_signatures.parquet missing stability columns")
    rows = []
    for (arm, domain), g in frame.groupby(["arm", "domain"], sort=True):
        if len(g) < min_patients: continue
        rows.append({"arm": arm, "domain": domain, "patient_count": len(g), "mean_same_domain_spearman": float(g.same_domain_spearman.mean()), "median_same_domain_spearman": float(g.same_domain_spearman.median()), "mean_best_other_spearman": float(g.best_other_domain_spearman.mean()), "mean_margin": float(g.margin.mean()), "positive_margin_fraction": float((g.margin > 0).mean())})
    return pd.DataFrame(rows)


# Small public aliases keep the mathematical contract easy to test/notebook.
patient_aware_signatures = patient_domain_signatures
aggregate_markers = aggregate_patient_signatures
compute_patient_aware_markers = top_positive_markers
compute_program_scores = score_programs
select_program_candidates = top_program_candidates
compute_hallmark_ora = hallmark_ora
deterministic_hungarian = hungarian_matching
best_matches = best_signature_matches
exact_prevalence_permutation = exact_prevalence_test
bootstrap_prevalence = prevalence_bootstrap
complete_prevalence = complete_patient_prevalence


def _load_anndata(path: Path) -> Any:
    try:
        import anndata as ad
    except ImportError as exc: raise RuntimeError("anndata is required on SLURM") from exc
    return ad.read_h5ad(path)


def _pick(columns: Iterable[str], names: tuple[str, ...], label: str) -> str:
    for n in names:
        if n in columns: return n
    raise ContractError(f"missing {label}")


def _safe_slide_id(value: Any) -> str:
    slide = str(value).strip()
    if not slide or slide in {".", ".."} or ".." in slide or "/" in slide or "\\" in slide or os.sep in slide or (os.altsep and os.altsep in slide):
        raise ContractError(f"unsafe slide identifier: {slide!r}")
    return slide


def _maps(frame: pd.DataFrame, output: Path, matching: pd.DataFrame, *, invalid_rows: int = N_INVALID) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise ContractError("matplotlib is required to render the 14-slide maps") from exc
    groups = list(frame.groupby("slide", sort=True))
    if len(groups) != 14 or any(group.empty for _, group in groups):
        raise ContractError("maps require exactly 14 nonempty slides")
    output_resolved = output.resolve()
    files = []; figures = []
    colors = {}
    legend_handles = []
    for i, r in matching.iterrows():
        colors[str(r.nmf_domain)] = i; colors[str(r.novae_domain)] = i
        from matplotlib.patches import Patch
        rho = "nan" if not np.isfinite(r.spearman) else f"{r.spearman:.3f}"
        legend_handles.append(Patch(facecolor=plt.cm.tab10((i + 1) / 9), label=f"NMF {r.nmf_domain} ↔ NOVAE {r.novae_domain} (rho={rho})"))
    for raw_slide, group in groups:
        slide = _safe_slide_id(raw_slide)
        path = (output / f"map_{slide}.png").resolve()
        if output_resolved not in path.parents: raise ContractError("map path escapes output directory")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, col, title in zip(axes, ["nmf_factor", "novae_domain"], ["NMF exploratory domain", "NOVAE exploratory domain"], strict=True):
            vals = group[col].astype(str).map(colors).fillna(-1); ax.scatter(group.x, group.y, c=vals, s=5, cmap="tab10", vmin=-1, vmax=8); ax.set_title(f"{title} — {slide} ({len(group)} shared valid spots)"); ax.set_aspect("equal"); ax.set_xlabel("preserved x"); ax.set_ylabel("preserved y")
        fig.suptitle(f"{slide}: {len(group)} shared valid spots | {invalid_rows} invalid NOVAE rows excluded cohort-wide; no imputation", fontsize=9)
        if legend_handles: fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.04))
        fig.savefig(path, dpi=180, bbox_inches="tight"); files.append(path.name); figures.append(fig); plt.close(fig)
        if not path.is_file() or path.stat().st_size == 0: raise ContractError(f"empty map output: {path}")
    if figures:
        pdf = output / "domain_maps.pdf"
        with PdfPages(pdf) as writer:
            # Figures are retained by PdfPages until savefig; recreate from PNGs
            for name in files:
                image = plt.imread(output / name); fig, ax = plt.subplots(figsize=(10, 4)); ax.imshow(image); ax.axis("off"); writer.savefig(fig); plt.close(fig)
        if not pdf.is_file() or pdf.stat().st_size == 0: raise ContractError("empty multipage map PDF")
        files.append(pdf.name)
    if len([name for name in files if name.endswith(".png")]) != 14: raise ContractError("expected 14 map PNGs")
    return files


def _validate_prior_inventory(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs: raise ContractError("prior validation output inventory is missing")
    resolved: dict[str, str] = {}
    for name, digest in outputs.items():
        name = str(name)
        if Path(name).name != name or name in {".", ".."} or not isinstance(digest, str) or len(digest) != 64: raise ContractError("unsafe or incomplete prior output inventory")
        path = (root / name).resolve()
        if root.resolve() not in path.parents or not path.is_file() or sha256(path) != digest: raise ContractError(f"prior output hash/path mismatch: {name}")
        resolved[name] = digest
    required = {"patient_logo_signatures.parquet", "graph_contract.json", "shared_observation_graph_contract.parquet"}
    if not required.issubset(resolved): raise ContractError("prior manifest must hash all consumed validation outputs")
    return resolved


def run_interpretation(novae_h5ad: str | Path, post_nmf_obs: str | Path, validation_dir: str | Path, output_dir: str | Path, *, expected_h5ad_sha256: str | None = EXPECTED_CALIBRATED_H5AD_SHA256, expected_post_sha256: str | None = EXPECTED_POST_NMF_OBS_SHA256, expected_valid: int = N_VALID, expected_invalid: int = N_INVALID, seed: int = DEFAULT_SEED, bootstrap_reps: int = DEFAULT_BOOTSTRAPS) -> Path:
    h5ad, post, validation_dir, output = map(Path, (novae_h5ad, post_nmf_obs, validation_dir, output_dir))
    output_resolved = output.resolve()
    for source in (h5ad, post, validation_dir):
        source_resolved = source.resolve()
        if output_resolved == source_resolved or output_resolved in source_resolved.parents or source_resolved in output_resolved.parents: raise ContractError("output path overlaps an input/report path")
    if output.exists(): raise ContractError(f"refusing existing output: {output}")
    if not h5ad.is_file() or not post.is_file() or not validation_dir.is_dir(): raise ContractError("required input/report is missing")
    validation_manifest = validation_dir / "manifest.json"; prior_logo = validation_dir / "patient_logo_signatures.parquet"; graph_contract = validation_dir / "graph_contract.json"; prior_observations = validation_dir / "shared_observation_graph_contract.parquet"
    if not validation_manifest.is_file() or not prior_logo.is_file() or not graph_contract.is_file() or not prior_observations.is_file(): raise ContractError("completed validation manifest, graph contract, shared observations, and patient_logo_signatures.parquet are required")
    h0, p0 = sha256(h5ad), sha256(post)
    if expected_h5ad_sha256 and h0 != expected_h5ad_sha256: raise ContractError("calibrated H5AD SHA256 mismatch")
    if expected_post_sha256 and p0 != expected_post_sha256: raise ContractError("post_nmf_obs SHA256 mismatch")
    validation_manifest_hash = sha256(validation_manifest)
    prior = json.loads(validation_manifest.read_text(encoding="utf-8")); prior_inputs = prior.get("inputs", {})
    if prior.get("contract") != "novae_spatial_biological_validation": raise ContractError("validation manifest is not the completed spatial-validation contract")
    prior_outputs = _validate_prior_inventory(validation_dir, prior)
    graph_info = json.loads(graph_contract.read_text(encoding="utf-8"))
    if graph_info.get("coordinates") != "identical shared x/y" or graph_info.get("expression_source") != "layers['counts']" or graph_info.get("nodes") != expected_valid: raise ContractError("validation graph/coordinate/count contract mismatch")
    novae_input = prior_inputs.get("novae_h5ad", {}); post_input = prior_inputs.get("post_nmf_obs", {})
    if not novae_input.get("path") or not post_input.get("path") or Path(str(novae_input.get("path"))).resolve() != h5ad.resolve() or Path(str(post_input.get("path"))).resolve() != post.resolve() or novae_input.get("sha256") != h0 or post_input.get("sha256") != p0: raise ContractError("validation manifest input path/hash mismatch")
    prior_artifact_hashes = {name: sha256(validation_dir / name) for name in prior_outputs}
    adata = _load_anndata(h5ad)
    if "counts" not in adata.layers: raise ContractError("layers['counts'] is required")
    if getattr(adata.layers["counts"], "shape", None) != (int(adata.n_obs), int(len(adata.var_names))): raise ContractError("layers['counts'] shape is not aligned")
    counts = _counts_array(adata.layers["counts"]); n = len(counts)
    if "spatial" not in adata.obsm: raise ContractError("obsm['spatial'] is required")
    spatial = np.asarray(adata.obsm["spatial"])
    try: spatial_numeric = spatial[:, :2].astype(float)
    except (TypeError, ValueError, IndexError) as exc: raise ContractError("obsm['spatial'] must be numeric") from exc
    if spatial.ndim != 2 or spatial.shape[0] != n or spatial.shape[1] < 2 or not np.isfinite(spatial_numeric).all(): raise ContractError("obsm['spatial'] must be finite n_obs by >=2")
    valid_key = "neighborhood_valid"; novae_key = "novae_domains_res1.0"
    if valid_key not in adata.obs or novae_key not in adata.obs: raise ContractError("NOVAE domain/validity columns missing")
    raw_valid = adata.obs[valid_key].tolist()
    if any(not isinstance(x, (bool, np.bool_)) for x in raw_valid): raise ContractError("neighborhood_valid must contain native booleans")
    valid = np.asarray(raw_valid, bool)
    if valid.sum() != expected_valid or (~valid).sum() != expected_invalid: raise ContractError("shared valid row count mismatch")
    ids = _text(adata.obs["unique_cell_id"] if "unique_cell_id" in adata.obs else adata.obs_names, "unique_cell_id")
    if len(set(ids)) != n: raise ContractError("unique_cell_id must be unique")
    obs_names = np.asarray([str(x) for x in adata.obs_names])
    if len(set(obs_names)) != n or not np.array_equal(ids, obs_names): raise ContractError("obs_names and unique_cell_id must align exactly")
    post_df = pd.read_csv(post); id_col = _pick(post_df.columns, ("unique_cell_id", "unique_cellid", "cell_id"), "post IDs"); factor_col = _pick(post_df.columns, ("NMF_factor", "nmf_factor", "dominant_nmf_factor"), "NMF factor")
    post_df["_id"] = _text(post_df[id_col], "post IDs"); by = post_df.set_index("_id")
    if set(by.index) != set(ids): raise ContractError("NMF/NOVAE unique_cell_id sets differ")
    nmf = by.loc[ids, factor_col].map(lambda x: str(int(float(x))) if float(x).is_integer() else str(x)).to_numpy()
    if set(nmf) != {str(i) for i in range(N_DOMAINS)}: raise ContractError("NMF factors must contain exactly 0-8")
    novae = np.asarray(adata.obs[novae_key].astype(str));
    if set(novae[valid]) != {f"L{i}" for i in range(N_DOMAINS)}: raise ContractError("valid NOVAE domains must contain exactly L0-L8")
    if (novae[~valid] != "").any() and not np.all(pd.isna(adata.obs.loc[~valid, novae_key])): raise ContractError("invalid NOVAE rows must have no domain assignment")
    disease_col = _pick(adata.obs.columns, ("Disease_State", "disease_state", "Disease/Health State"), "disease"); patient_col = _pick(adata.obs.columns, ("patient", "Patient", "subject"), "patient"); slide_col = _pick(adata.obs.columns, ("sample_id", "slide", "library_id"), "slide")
    patients = _text(adata.obs[patient_col], "patients"); disease = _text(adata.obs[disease_col], "disease"); slides = _text(adata.obs[slide_col], "slides")
    if len(set(patients)) != 14 or len(set(slides)) != 14: raise ContractError("interpretation requires exactly 14 patients and 14 slides")
    disease_by_patient = pd.DataFrame({"patient": patients, "disease": disease}).drop_duplicates()
    if disease_by_patient.patient.nunique() != 14 or disease_by_patient.groupby("patient").disease.nunique().ne(1).any() or disease_by_patient.disease.value_counts().to_dict() != {"healthy": 4, "systemic_sclerosis": 10}: raise ContractError("cohort must contain 4 healthy and 10 systemic_sclerosis patients")
    post_patient = _pick(by.columns, ("patient", "Patient", "subject"), "post patient"); post_disease = _pick(by.columns, ("Disease_State", "disease_state", "Disease/Health State"), "post disease")
    if not np.array_equal(_text(by.loc[ids, post_patient], "post patients"), patients) or not np.array_equal(_text(by.loc[ids, post_disease], "post disease"), disease): raise ContractError("post_nmf_obs patient/disease metadata disagrees with H5AD")
    for key in ("sample_id", "slide"):
        if key in by.columns and key in adata.obs.columns and not np.array_equal(_text(by.loc[ids, key], f"post {key}"), _text(adata.obs[key], f"H5AD {key}")): raise ContractError(f"post_nmf_obs {key} metadata disagrees with H5AD")
    post_slide_key = _pick(by.columns, ("sample_id", "slide", "library_id"), "post sample/slide")
    if not np.array_equal(_text(by.loc[ids, post_slide_key], "post sample/slide"), slides): raise ContractError("post_nmf_obs sample/slide metadata disagrees with H5AD")
    genes = np.asarray([str(x) for x in adata.var_names]); order = np.flatnonzero(valid)
    preserved = pd.read_parquet(prior_observations)
    if not {"unique_cell_id", "x", "y"}.issubset(preserved.columns): raise ContractError("validation shared observations lack coordinates")
    current_coords = pd.DataFrame({"unique_cell_id": ids[order], "x": np.asarray(adata.obsm["spatial"])[order, 0], "y": np.asarray(adata.obsm["spatial"])[order, 1]})
    check_coords = preserved[["unique_cell_id", "x", "y"]].sort_values("unique_cell_id").reset_index(drop=True)
    if not np.array_equal(check_coords.unique_cell_id.to_numpy(), current_coords.sort_values("unique_cell_id").unique_cell_id.to_numpy()) or not np.allclose(check_coords[["x", "y"]], current_coords.sort_values("unique_cell_id")[["x", "y"]]): raise ContractError("coordinates differ from completed validation")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        all_sigs = []; all_agg = []
        for arm, labels in (("nmf", nmf), ("novae", novae)):
            sig = patient_domain_signatures(counts[order], labels[order], patients[order], eligible_detected_genes(counts[order]), arm=arm, gene_names=genes)
            agg = aggregate_patient_signatures(sig); all_sigs.append(sig); all_agg.append(agg)
        full = pd.concat(all_sigs, ignore_index=True); agg = pd.concat(all_agg, ignore_index=True); top = top_positive_markers(agg)
        programs = score_programs(agg); top_programs = top_program_candidates(programs); ora = hallmark_ora(top, [genes[i] for i in eligible_detected_genes(counts[order])])
        nmf_agg, novae_agg = agg[agg.arm == "nmf"], agg[agg.arm == "novae"]; sim = signature_similarity(nmf_agg, novae_agg); matching = hungarian_matching(sim); best_matches = best_signature_matches(sim)
        contingency, agreement = spot_agreement(nmf[order], novae[order], patients[order]); overlap = matched_overlap(nmf[order], novae[order], patients[order], matching)
        coverage = pd.DataFrame({"patient": patients[order], "disease": disease[order], "nmf_factor": nmf[order], "novae_domain": novae[order]})
        prevalence = complete_patient_prevalence(coverage)
        prev_frames = []
        for arm, g in prevalence.groupby("arm", sort=True):
            x = g.rename(columns={"arm": "_arm"})
            t = exact_prevalence_test(x, disease_col="disease", value_col="proportion"); t.insert(0, "arm", arm); prev_frames.append(t)
        prev_tests = pd.concat(prev_frames, ignore_index=True); prev_ci = pd.concat([prevalence_bootstrap(g, reps=bootstrap_reps, seed=seed).assign(arm=arm) for arm, g in prevalence.groupby("arm", sort=True)], ignore_index=True)
        prevalence_summary = prevalence.groupby(["arm", "domain", "disease"], as_index=False)["proportion"].mean().rename(columns={"proportion": "mean_patient_proportion"})
        stability = stability_summary(prior_logo)
        observations = pd.DataFrame({"unique_cell_id": ids[order], "slide": slides[order], "patient": patients[order], "disease": disease[order], "x": spatial_numeric[order, 0], "y": spatial_numeric[order, 1], "nmf_factor": nmf[order], "novae_domain": novae[order], "novae_valid": True})
        for frame, name in ((observations, "shared_observations.parquet"), (prevalence, "patient_domain_prevalence.parquet"), (full, "patient_aware_markers_full.parquet"), (agg, "patient_aware_markers_aggregated.parquet"), (top, "patient_aware_markers_top100.csv"), (programs, "fixed_program_scores.csv"), (top_programs, "fixed_program_top_candidates.csv"), (ora, "hallmark_ora.csv"), (sim, "nmf_novae_similarity.csv"), (matching, "nmf_novae_matching.csv"), (best_matches, "best_signature_matches.csv"), (contingency, "spot_contingency.csv"), (agreement, "spot_agreement.csv"), (overlap, "matched_overlap.csv"), (prev_tests, "exact_prevalence_tests.csv"), (prev_ci, "prevalence_bootstrap_ci.csv"), (prevalence_summary, "disease_prevalence_summary.csv"), (stability, "patient_logo_stability_summary.csv")):
            _atomic_table(frame, stage / name)
        map_files = _maps(observations, stage, matching, invalid_rows=int((~valid).sum()))
        h1, p1 = sha256(h5ad), sha256(post)
        if h1 != h0 or p1 != p0: raise ContractError("input changed while interpretation was running")
        if any(sha256(validation_dir / name) != digest for name, digest in prior_artifact_hashes.items()) or sha256(validation_manifest) != validation_manifest_hash: raise ContractError("prior validation artifact changed while interpretation was running")
        program_resource = _resource_path(PROGRAM_RESOURCE); hallmark_resource = _resource_path(HALLMARK_RESOURCE); resource_manifest = _resource_path(RESOURCE_MANIFEST)
        resource_hashes = {PROGRAM_RESOURCE: sha256(program_resource), HALLMARK_RESOURCE: sha256(hallmark_resource), HALLMARK_SIDECAR: sha256(_resource_path(HALLMARK_SIDECAR)), RESOURCE_MANIFEST: sha256(resource_manifest)}
        if resource_hashes[PROGRAM_RESOURCE] != PROGRAM_SHA256 or resource_hashes[HALLMARK_RESOURCE] != HALLMARK_SHA256: raise ContractError("fixed resource checksum mismatch")
        sidecar = _resource_path(HALLMARK_RESOURCE + ".sha256")
        if sidecar.read_text(encoding="utf-8").strip() != f"{HALLMARK_SHA256}  {hallmark_resource.name}": raise ContractError("Hallmark checksum sidecar mismatch")
        hallmark_sets = parse_gmt(hallmark_resource)
        if len(hallmark_sets) != HALLMARK_SET_COUNT or any(not members for members in hallmark_sets.values()): raise ContractError("Hallmark GMT set inventory mismatch")
        resource_info = json.loads(resource_manifest.read_text(encoding="utf-8"))
        hallmark_info = resource_info.get("hallmark_human_2025.1", {})
        if hallmark_info.get("sha256") != HALLMARK_SHA256 or hallmark_info.get("url") != "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt" or hallmark_info.get("source") != "Broad Institute MSigDB 2025.1.Hs public release" or hallmark_info.get("license") != "CC BY 4.0" or hallmark_info.get("license_url") != "https://gsea-msigdb.org/gsea/msigdb_license_terms.jsp" or resource_info.get("skin_marker_programs", {}).get("file") != Path(PROGRAM_RESOURCE).name or resource_info.get("skin_marker_programs", {}).get("sha256") != PROGRAM_SHA256: raise ContractError("resource manifest attribution/checksum mismatch")
        summary = {"contract": "novae_skin_domain_interpretation", "status": "exploratory", "K": 9, "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()), "patients": int(len(set(patients))), "disease_used_for_marker_selection": False, "raw_expression_source": "layers['counts']", "no_imputation": True, "prior_validation_manifest": str(validation_manifest), "maps": map_files, "global_agreement": agreement.loc[agreement.group == "global"].to_dict("records"), "matched_similarity_summary": matching[["nmf_domain", "novae_domain", "spearman"]].to_dict("records"), "best_signature_matches": best_matches.to_dict("records"), "hallmark_q_lt_0_05_count": int((ora.q_value < 0.05).sum()) if not ora.empty else 0, "prevalence_effect_domains": prev_tests[["arm", "domain", "observed_effect_ssc_minus_healthy"]].to_dict("records"), "blocked_status": {"automatic_cell_type_labels": "blocked", "confirmatory_claims": "blocked", "histology_overlay": "blocked"}, "circularity": ["NMF factors and NOVAE domains are compared descriptively; marker/program resources do not select domains", "reference=all and cohort-derived domains preclude confirmatory interpretation"], "limitations": ["domain names are not cell-type ground truth", "cohort-derived/reference=all exploratory", "prevalence tests are descriptive and patient-level", "NMF/NOVAE matching is only for descriptive alignment/maps"]}
        _atomic_json(summary, stage / "machine_summary.json")
        _atomic_table(pd.DataFrame([{"metric": "valid_rows", "value": int(valid.sum())}, {"metric": "invalid_rows", "value": int((~valid).sum())}, {"metric": "patients", "value": int(len(set(patients)))}, {"metric": "marker_selection_disease_blind", "value": True}]), stage / "machine_summary.csv")
        manifest = {"contract": "novae_skin_domain_interpretation", "inputs": {"novae_h5ad": {"path": str(h5ad), "sha256_before": h0, "sha256_after": h1}, "post_nmf_obs": {"path": str(post), "sha256_before": p0, "sha256_after": p1}, "validation_manifest": {"path": str(validation_manifest), "sha256_before": validation_manifest_hash, "sha256_after": sha256(validation_manifest)}, "prior_validation_outputs": prior_artifact_hashes}, "resources": resource_hashes, "outputs": {}, "code_sha256": sha256(Path(__file__))}
        manifest["outputs"] = {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}; _atomic_json(manifest, stage / "manifest.json")
        if output.exists(): raise ContractError(f"refusing output created during run: {output}")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--novae-h5ad", required=True, type=Path); p.add_argument("--post-nmf-obs", required=True, type=Path); p.add_argument("--validation-dir", required=True, type=Path); p.add_argument("--output-dir", required=True, type=Path); p.add_argument("--seed", type=int, default=DEFAULT_SEED); p.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAPS); return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv); run_interpretation(a.novae_h5ad, a.post_nmf_obs, a.validation_dir, a.output_dir, seed=a.seed, bootstrap_reps=a.bootstrap_reps)
    except (ContractError, RuntimeError, OSError, ValueError) as exc:
        print(f"interpretation blocked: {exc}", file=os.sys.stderr); return 2
    return 0

if __name__ == "__main__": raise SystemExit(main())
