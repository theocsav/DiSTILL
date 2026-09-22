# NOVAE downstream pseudo-FOV contract audit

The historical 164-row and full-sweep 225-row skin contracts are an evidence
question, not a model-selection question. The read-only audit is
`scripts/audit_novae_downstream_contract.py`; real H5AD/table reads are allowed
only in its one-CPU, 64 GB SLURM job:

```bash
scripts/submit_novae_downstream_contract_audit.sh --render-only
scripts/submit_novae_downstream_contract_audit.sh
```

Defaults are the fixed HPG source/run paths in the launcher and Python parser.
Synthetic path overrides are intended for tests and controlled HPG reruns. The
launcher validates scheduler/path values, does not process data on a login
node, uses the local NOVAE conda environment, requests no GPU, and refuses an
existing output directory. The Python audit opens both H5ADs with
`backed="r"`, never writes an input, and publishes its JSON and CSV bundle by
atomic directory replacement.

## Gate

For each candidate, the audit separately inventories artifact presence and
contract acceptance. It reports missing `cosmx_with_nmf.h5ad`, `post_nmf_obs.csv`,
enrichment and niche-gene tables, `MLP_FOVFeatures_inputs/combined_features_filtered.parquet`,
`targets_y.parquet`, `groups.parquet`, and optional legacy provenance/manifest
artifacts. It validates explicit disease-state casing (`Disease_State` or the
recorded lowercase compatibility key), patient/FOV mapping, unique cell IDs,
source/NMF observation order and sets, NMF columns, row preservation, one
patient/label per FOV, exact feature/target/group indices and values, numeric
finite features, duplicates, missingness, and count summaries. Composition is
reconstructed exactly as `build_fov_classifier_inputs.py` does it: an
`NMF_factor` crosstab with sorted FOV index and `nmf_prop_*` columns. The
combined table's row order is the frozen canonical downstream index; target and
group indices and values must match it exactly. Raw enrichment and niche-gene
tables must cover that index but may retain producer order and extra rows, which
are recorded rather than treated as failures. Excluded post-NMF FOVs are
recorded explicitly. The comparison file records exact FOV intersections and
differences only; it does not infer an outcome-driven reason for 164 versus
225.

Acceptance requires every required check to pass. A valid completed historical
164 contract is the preferred frozen comparison. The full-sweep contract is
eligible only when independently complete and valid. Classifier performance
cannot choose between them. Until this gate has evidence, primary NOVAE
integration remains blocked.

A selected adapter must freeze the exact feature-column index, target values,
and patient-group index from this audit. It must join calibrated NOVAE labels
by preserved `obs`/`unique_cell_id`, never by retiling or reconstructing FOVs.
