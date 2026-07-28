# Pipeline Stage Review

Status: active
Last reviewed: 2026-07-28
Companion to: [MLP_EVALUATION_AND_LEAKAGE.md](MLP_EVALUATION_AND_LEAKAGE.md)

Review of the non-MLP stages: `cell2loc`, `nmf`, `post_nmf`, `rcausal_mgm`, and the
API preflight in `apps/api/app/validation.py`. The MLP stage is covered separately.

Findings are ordered by how much they affect a reported result.

---

## Summary

| Stage | Verdict |
|---|---|
| `nmf` k-selection | **Sound.** Label-free criteria, no disease leakage |
| `nmf` reproducibility | **Caveat.** Two non-equivalent backends; see 2.1 |
| `post_nmf` statistics | **Sound.** BH-FDR applied and enforced |
| `post_nmf` -> MLP handoff | **Sound.** Unfiltered feature pool passed downstream |
| `rcausal_mgm` sample independence | **Serious.** Pseudo-replication; see 3.0 |
| `rcausal_mgm` reproducibility | **Fixed.** Seeds added; see 3.1 |
| `rcausal_mgm` reported figure | **Open.** Renders the unstabilized graph; see 3.2 |
| `validation.py` preflight | **Sound.** |

---

## 1. What is correct

Recorded explicitly so it is not "simplified" away later.

**NMF k-selection uses no disease labels.** `nmf_selection_method` supports
`fixed_k`, `elbow_k`, `poisson_redundancy_k`, and
`poisson_cumulative_improvement_k`. The Poisson methods select k by factor
redundancy (maximum pairwise cosine similarity between factor loadings) or by
cumulative reconstruction improvement. Both are unsupervised. `Disease_State`
appears nowhere in the k-selection path. The `reference_label_key` referenced during
cell2location setup is a **cell-type** annotation on the reference, not a disease
label.

**NMF seeds are swept, not cherry-picked on an outcome.** For each k the script runs
`poisson_n_runs` seeds and keeps the one with the lowest factor redundancy. That is
unsupervised model selection, which is legitimate.

**Both NMF backends check convergence.** sklearn uses `tol=1e-4`; the torch path
checks relative error change of `1e-5` with a patience counter rather than running a
fixed iteration count.

**Post-NMF multiple testing is handled.** `IBD_Post_NMF_Analysis.ipynb` applies
Benjamini-Hochberg via `multipletests(..., method="fdr_bh")` and filters at
`p_adj < 0.05` for both enrichment and niche-gene features.

**The post-NMF to MLP handoff does not leak.** The notebook computes FDR-filtered
`selected_enrichment_features` for its own reporting, but the parquet files consumed
by the MLP (`enrichment_features_fov.parquet`, `niche_gene_features_fov.parquet`)
are written from the *unfiltered* column lists. The only thing the MLP takes from
`combined_features_filtered.parquet` is the `nmf_prop_*` columns, which are
unsupervised. So the MLP's per-fold selection operates on a full candidate pool that
was never filtered using disease labels.

**Preflight validation is sound.** `apps/api/app/validation.py` checks join-key
strategy resolution, path containment within `ARTIFACT_ROOTS`, stage data contracts,
and missing/extra metadata row thresholds. No issues found.

---

## 2. NMF

### 2.1 The two backends are not numerically equivalent

`nmf_backend` selects between sklearn and a hand-written torch implementation. They
differ in ways that produce different factorizations for the same `k` and `seed`:

| | sklearn | torch |
|---|---|---|
| Initialisation | `init='nndsvda'` (deterministic SVD-based) | `torch.rand` uniform |
| dtype | float64 | float64 on CPU, **float32 on CUDA** |
| Updates | sklearn `mu` solver | hand-written multiplicative updates |

NMF is non-convex, so a different initialisation converges to a different local
optimum. This is not a bug - both are valid NMF - but it has consequences:

- A run produced with `nmf_backend: torch, nmf_device: cuda` **cannot be reproduced**
  with the default sklearn path, or on CPU (float32 vs float64).
- Niches are not comparable across runs that used different backends.
- 6 presets set `torch`/`cuda`; the script default is `sklearn`.

Because niches feed post-NMF features, the MLP, the causal graphs, and the
manuscript figures, the backend is part of the provenance of every downstream claim.

**Mitigation.** The patched script copied into each run directory has the resolved
`nmf_backend` and `nmf_device` literals baked in, so provenance is recoverable from
the run directory. What is *not* recorded is the resolved device when
`nmf_device: auto` (which becomes cuda or cpu depending on the node) or the resulting
dtype.

**Recommendation.** Set `nmf_device` explicitly rather than `auto` for anything
reported, and state the backend alongside k in the methods section.

### 2.2 NMF is fit on all samples

The factorization is computed over every FOV, including those that later become MLP
test folds. No labels are involved, so this is not label leakage, and transductive
unsupervised feature extraction is common practice. It should still be disclosed:
the representation the classifier consumes was estimated with the test FOVs present.
A fully inductive design would refit NMF inside each outer fold, at substantial
compute cost.

---

## 3. Causal discovery (`rcausal_mgm`)

### 3.0 Pseudo-replication: FOVs treated as independent samples - open

This is the most serious statistical issue found in the pipeline.

`pipeline_assets/IBD_RCausalMGM_Preparation.py` builds the analysis table keyed by
FOV:

```python
df["field_of_view"] = df["patient"].astype(str) + "_" + df["fov"].astype(str)
```

`patient` is used only to construct that key and is then dropped. The R scripts do
`column_to_rownames("field_of_view")` and run `fciStable` over the resulting rows,
with `field_of_view` explicitly excluded from the variable set. There is no patient
or subject variable in the model and no adjustment for clustering.

So for the skin cohort, FCI runs on **164 FOV rows as though they were 164
independent observations, when the effective sample size is 14 patients**.

This matters most for the disease edges. `Disease.Health.State` is a *patient-level*
attribute: every FOV from one patient carries an identical label. The conditional
independence tests backing any edge into disease state are therefore computed at
roughly 12x the true sample size, which makes them strongly anti-conservative.
Expect spurious edges and over-confident orientation.

The inconsistency is the clearest signal that something is wrong. On the same data:

- the MLP stage groups scrupulously by patient (`LeaveOneGroupOut`, no FOV spanning
  folds)
- the causal stage ignores patient structure entirely

Both cannot be right about what constitutes an independent sample.

**Not changed here.** The fix is a modelling decision with real tradeoffs at n = 14,
and it should be made deliberately rather than by a code edit. The options:

1. **Aggregate to patient level.** Statistically clean, but drops to n = 14
   observations, which is likely too few for FCI orientation.
2. **Keep FOV rows, add patient as a variable** so the graph can condition on it.
   Retains resolution, but patient is a high-cardinality nuisance variable.
3. **Use a clustered or permutation-based independence test** that respects patient
   blocks. Most faithful, most work.
4. **Report FOV-level graphs as exploratory only**, with disease edges qualified,
   and treat patient-level aggregation as the confirmatory analysis.

Option 4 is the cheapest honest position if the graphs are already in a draft.

This is worth raising directly with whoever owns the causal methodology, since the
choice depends on what the graphs are being used to claim.

### 3.1 No random seed - fixed

`fciStable` and `bootstrap` are both stochastic, and none of the three R scripts
called `set.seed()`. Neither the causal graph nor the bootstrap stabilities were
reproducible run to run.

All three scripts now accept `--seed` (default 42) and call `set.seed(seed)` before
each `fciStable` call, seeded per target so that target ordering does not shift later
targets' results:

- `pipeline_assets/rCausalMGM_Rscript_NicheComposition.R`
- `pipeline_assets/rCausalMGM_Rscript_NeighborhoodInteractions.R`
- `pipeline_assets/rCausalMGM_Rscript_NicheGeneFeatures.R`

**Note:** existing causal figures were produced without a seed and cannot be
reproduced exactly, even now. Re-running will produce a graph that is reproducible
going forward but not necessarily identical to what was previously rendered.

### 3.2 The rendered figure is the unstabilized graph - open

Each script does the right analysis and then renders the wrong artifact.

`rCausalMGM_Rscript_NicheComposition.R` lines ~154-182:

```r
fci_graph   <- fciStable(data = subset_data, orientRule = "maxp", alpha = 0.05, ...)
boot_results <- bootstrap(data = subset_data, graph = fci_graph, numBoots = num_boots, ...)
saveGraph(fci_graph,   filename = fci_sif_path)
saveGraph(boot_results, filename = ..._bootstrap_rCausalMGM.sif)
write.csv(boot_results$stabilities, ..._bootstrap_stabilities.csv)
...
write_clean_dot_from_sif(fci_sif_path, dot_path, "Niche Disease State Graph")   # <- raw FCI
```

The polished PNG (`rcausal_graphviz_niche_disease_state.png`) - identified in
`docs/SKIN_KIDNEY_MLP_FINDINGS_2026-07-03.md` as the canonical presentation source -
is built from `fci_sif_path`, the **single-run FCI graph**. The bootstrap ensemble
and its per-edge stabilities are computed and saved but are not used to filter the
edges that get drawn. The ensemble graph *is* plotted into the combined PDF, so the
information exists; it just is not what the presentation figure shows.

This is the same shape of problem as the MLP issue: the rigorous artifact is
produced, and the reported artifact is not derived from it.

**Not changed here**, because switching the rendered graph would change published
figures and that is an author decision.

**Recommendation.** Either render from the bootstrap ensemble SIF, or annotate edges
with their bootstrap stability and state the threshold in the caption. Any edge
reported as a finding should carry its stability value.

### 3.3 `numBoots = 20` is low

Twenty bootstrap replicates gives a coarse stability estimate (resolution 0.05 per
edge). If stabilities are going to be reported or used for filtering, raise this.
Left at the existing default so current behaviour is unchanged; override with
`--num-boots`.

---

## 4. cell2location

No correctness issues found in the wiring. Training epochs, accelerator, and devices
are preset-driven and recorded in the patched script in the run directory. The
reference model is trained on cell-type annotations, and the resulting `inf_aver`
signature feeds spatial mapping as expected.

Cell-type assignment quality itself was not evaluated - that requires the reference
data and is a biological validation question rather than a code-correctness one.

---

## 5. Open items

1. **Decide how to handle patient clustering in the causal stage (3.0).** Highest
   priority: it affects every disease-linked edge.
2. Decide whether causal figures should render from the bootstrap ensemble (3.2).
3. Pin `nmf_device` explicitly for reported runs; record resolved device and dtype
   in run provenance (2.1).
4. Disclose transductive NMF in methods (2.2).
5. Raise `--num-boots` if stabilities will be reported (3.3).
