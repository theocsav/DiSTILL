# Skin 250um min5 exploratory arm

This arm tests whether the 250um pseudo-FOV classification can run when FOV
filtering retains FOVs with at least 5 total cells. It reuses the completed
250um NMF output and writes to a separate `fullsweep_min5` output directory.

The neighborhood-enrichment implementation still requires at least 2 cells
per FOV with finite spatial coordinates and finite area-derived diameters.
This is an exploratory classification check and is not a replacement for the
primary 500um and 750um analyses.

Submit from HiPerGator:

```bash
bash scripts/submit_skin_pseudofov_250um_min5.sh
```

The chain is `post_nmf -> mlp_tune_once -> mlp_eval_fixed/report`. It does not
rerun cell2location or NMF.
