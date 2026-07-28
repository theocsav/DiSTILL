#!/usr/bin/env bash
# Submit an evaluation-protocol comparison for one run's MLP feature tables.
#
# Measures how much the discontinued search-then-report-on-the-same-folds
# protocol inflates a result, against an honest nested estimate, over identical
# features/targets/groups. See docs/MLP_EVALUATION_AND_LEAKAGE.md section 6b.
#
# Usage:
#   scripts/submit_protocol_comparison.sh <run_outputs_dir> [label] [logo|sgkf]
#
# Example:
#   scripts/submit_protocol_comparison.sh \
#     /blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs \
#     skin logo
set -euo pipefail

REPO_DIR="/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"
CONDA_ENV="/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4"
ACCOUNT="kejun.huang"
QOS="kejun.huang-b"

RUN_OUT="${1:?usage: $0 <run_outputs_dir> [label] [logo|sgkf]}"
LABEL="${2:-protocmp}"
OUTER_CV="${3:-logo}"

# The FOV-level tables the leakage-safe MLP consumes; fall back to the run root.
if [[ -f "${RUN_OUT}/MLP_FOVFeatures_inputs/combined_features_filtered.parquet" ]]; then
  FEAT_DIR="${RUN_OUT}/MLP_FOVFeatures_inputs"
else
  FEAT_DIR="${RUN_OUT}"
fi

for f in combined_features_filtered.parquet targets_y.parquet groups.parquet; do
  if [[ ! -f "${FEAT_DIR}/${f}" ]]; then
    echo "ERROR: missing ${FEAT_DIR}/${f}" >&2
    echo "Checked ${RUN_OUT}/MLP_FOVFeatures_inputs and ${RUN_OUT}." >&2
    exit 1
  fi
done

OUTDIR="${RUN_OUT}/ProtocolComparison_${LABEL}"
LOG_DIR="${REPO_DIR}/runs/protocol_comparison_logs"
mkdir -p "${LOG_DIR}"
JOB_SCRIPT="${LOG_DIR}/submit_${LABEL}.sh"

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=protocmp_${LABEL}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${LOG_DIR}/protocmp_${LABEL}_%j.out
#SBATCH --error=${LOG_DIR}/protocmp_${LABEL}_%j.err
set -euo pipefail

module load conda
set +u
conda activate ${CONDA_ENV}
set -u

cd ${REPO_DIR}
python scripts/compare_evaluation_protocols.py \\
  --features ${FEAT_DIR}/combined_features_filtered.parquet \\
  --targets  ${FEAT_DIR}/targets_y.parquet \\
  --groups   ${FEAT_DIR}/groups.parquet \\
  --outdir   ${OUTDIR} \\
  --outer-cv ${OUTER_CV} \\
  --grid compact \\
  --jobs \${SLURM_CPUS_PER_TASK:-8}
EOF

chmod +x "${JOB_SCRIPT}"
echo "features : ${FEAT_DIR}"
echo "outdir   : ${OUTDIR}"
echo "job      : ${JOB_SCRIPT}"
sbatch "${JOB_SCRIPT}"
