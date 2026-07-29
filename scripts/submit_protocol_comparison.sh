#!/usr/bin/env bash
# Submit an evaluation-protocol comparison for one run's MLP feature tables.
#
# Measures how much the discontinued search-then-report-on-the-same-folds
# protocol inflates a result, against an honest nested estimate, over identical
# features/targets/groups. See docs/MLP_EVALUATION_AND_LEAKAGE.md section 6b.
#
# Usage:
#   scripts/submit_protocol_comparison.sh <run_outputs_dir> [label] [logo|sgkf] [features_dir]
#
# features_dir is optional and overrides discovery. Pass it whenever a run has
# more than one candidate feature directory, so the choice is explicit rather
# than inferred.
#
# Example:
#   scripts/submit_protocol_comparison.sh \
#     /blue/.../runs/skin_visium_ssc_1mmfov_k4_generalization/outputs \
#     skin_k4 logo \
#     /blue/.../runs/skin_visium_ssc_1mmfov_k4_generalization/outputs/MLP_FOVFeatures_inputs_k4
set -euo pipefail

REPO_DIR="/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"
CONDA_ENV="/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4"
ACCOUNT="kejun.huang"
QOS="kejun.huang-b"

RUN_OUT="${1:?usage: $0 <run_outputs_dir> [label] [logo|sgkf] [features_dir]}"
LABEL="${2:-protocmp}"
OUTER_CV="${3:-logo}"
FEAT_DIR_OVERRIDE="${4:-}"

# Resolve the feature directory. Runs can carry several MLP_FOVFeatures_inputs*
# directories (one per configuration), and the run root holds a patient-level
# table from post_nmf. Silently falling back to the root would evaluate the wrong
# matrix and still look like a clean result, so ambiguity is a hard error.
if [[ -n "${FEAT_DIR_OVERRIDE}" ]]; then
  FEAT_DIR="${FEAT_DIR_OVERRIDE}"
else
  mapfile -t CANDIDATES < <(
    find "${RUN_OUT}" -maxdepth 1 -type d -name 'MLP_FOVFeatures_inputs*' \
      -exec test -f '{}/combined_features_filtered.parquet' \; -print | sort
  )
  case "${#CANDIDATES[@]}" in
    1) FEAT_DIR="${CANDIDATES[0]}" ;;
    0)
      FEAT_DIR="${RUN_OUT}"
      echo "NOTE: no MLP_FOVFeatures_inputs* directory found; using the run root." >&2
      echo "      That table is usually patient-level, not FOV-level. Verify it." >&2
      ;;
    *)
      echo "ERROR: several feature directories under ${RUN_OUT}:" >&2
      printf '  %s\n' "${CANDIDATES[@]}" >&2
      echo "Pass one explicitly as the 4th argument." >&2
      exit 1
      ;;
  esac
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
