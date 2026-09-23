#!/usr/bin/env bash
# Render or submit the predeclared historical-164 classifier ablation.
set -euo pipefail

RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then RENDER_ONLY=1; shift; fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--render-only|--no-submit]" >&2; exit 2; }

REPO_DIR="${NOVAE_CLASSIFIER_ABLATION_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
NMF_FEATURE_DIR="${NOVAE_CLASSIFIER_ABLATION_NMF_FEATURE_DIR:-${REPO_DIR}/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_inputs}"
NMF_SOURCE_DIR="${NOVAE_CLASSIFIER_ABLATION_NMF_SOURCE_DIR:-${REPO_DIR}/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs}"
NOVAE_FEATURE_DIR="${NOVAE_CLASSIFIER_ABLATION_NOVAE_FEATURE_DIR:-${REPO_DIR}/runs/novae_res1_fov_features_historical164_typed_v2/features}"
FULL_PRIMARY_DIR="${NOVAE_CLASSIFIER_ABLATION_FULL_PRIMARY_DIR:-${REPO_DIR}/runs/novae_nmf_comparison_20260922T214308Z_1310236/historical164_nmf_vs_novae}"
RUN_ROOT="${NOVAE_CLASSIFIER_ABLATION_RUN_ROOT:-${REPO_DIR}/runs/novae_classifier_ablation_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUTPUT_ROOT="${NOVAE_CLASSIFIER_ABLATION_OUTPUT_ROOT:-${RUN_ROOT}/historical164_classifier_ablation}"
JOB_SCRIPT="${NOVAE_CLASSIFIER_ABLATION_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_classifier_ablation.sbatch}"
LOG_DIR="${NOVAE_CLASSIFIER_ABLATION_LOG_DIR:-${RUN_ROOT}/logs}"
CONDA_ENV="${NOVAE_CLASSIFIER_ABLATION_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"
QOS="${NOVAE_QOS:-kejun.huang}"
PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_CLASSIFIER_ABLATION_TIME:-24:00:00}"
JOB_NAME="${NOVAE_CLASSIFIER_ABLATION_JOB_NAME:-novae_cls_ablate}"

for name in NICHERUNNER_MLP_UNIT NICHERUNNER_MLP_MODE NICHERUNNER_MLP_BACKEND NICHERUNNER_MLP_GRID_PROFILE NICHERUNNER_MLP_SELECTION_METRIC NICHERUNNER_MLP_RESAMPLING NICHERUNNER_MLP_DECISION_THRESHOLD NICHERUNNER_MLP_MAX_EPOCHS NICHERUNNER_MLP_PATIENCE NICHERUNNER_TOP_ENRICHMENT_FEATURES NICHERUNNER_TOP_NICHE_GENE_FEATURES NICHERUNNER_SKIP_SHAP NICHERUNNER_MLP_DEVICE NICHERUNNER_COMPOSITION_PREFIX NICHERUNNER_MLP_FIXED_PARAMS_PATH NICHERUNNER_MLP_BEST_PARAMS_OUT NICHERUNNER_OUTPUT_DIR NICHERUNNER_SOURCE_OUTPUT_DIR NICHERUNNER_MLP_OUTPUT_DIR; do
  [[ -z "${!name-}" ]] || { echo "inherited ${name} is forbidden by frozen classifier-ablation protocol" >&2; exit 2; }
done

safe_path() { [[ -n "$2" && "$2" != *$'\n'* && "$2" != *$'\r'* && "$2" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "$1 contains unsafe path characters" >&2; exit 2; }; }
safe_value() { [[ -n "$2" && "$2" != *$'\n'* && "$2" != *$'\r'* && "$2" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "$1 contains unsafe scheduler characters" >&2; exit 2; }; }
for pair in "REPO_DIR:${REPO_DIR}" "NMF_FEATURE_DIR:${NMF_FEATURE_DIR}" "NMF_SOURCE_DIR:${NMF_SOURCE_DIR}" "NOVAE_FEATURE_DIR:${NOVAE_FEATURE_DIR}" "FULL_PRIMARY_DIR:${FULL_PRIMARY_DIR}" "RUN_ROOT:${RUN_ROOT}" "OUTPUT_ROOT:${OUTPUT_ROOT}" "JOB_SCRIPT:${JOB_SCRIPT}" "LOG_DIR:${LOG_DIR}" "CONDA_ENV:${CONDA_ENV}"; do safe_path "${pair%%:*}" "${pair#*:}"; done
for pair in "ACCOUNT:${ACCOUNT}" "QOS:${QOS}" "TIME_LIMIT:${TIME_LIMIT}" "JOB_NAME:${JOB_NAME}"; do safe_value "${pair%%:*}" "${pair#*:}"; done
[[ "${TIME_LIMIT}" =~ ^(1[2-9]|2[0-4]):[0-5][0-9]:[0-5][0-9]$ ]] || { echo "TIME_LIMIT must be between 12:00:00 and 24:59:59" >&2; exit 2; }
[[ -d "${REPO_DIR}" && -f "${REPO_DIR}/scripts/run_novae_classifier_ablation.py" ]] || { echo "repository/orchestrator not found" >&2; exit 2; }
if [[ -e "${OUTPUT_ROOT}" ]]; then
  [[ -d "${OUTPUT_ROOT}" ]] || { echo "output root is not a directory: ${OUTPUT_ROOT}" >&2; exit 2; }
  [[ ! -e "${OUTPUT_ROOT}/aggregate" ]] || { echo "refusing published aggregate: ${OUTPUT_ROOT}/aggregate" >&2; exit 2; }
fi
[[ ! -e "${JOB_SCRIPT}" ]] || { echo "refusing existing job script: ${JOB_SCRIPT}" >&2; exit 2; }
for input_dir in "${NMF_FEATURE_DIR}" "${NMF_SOURCE_DIR}" "${NOVAE_FEATURE_DIR}" "${FULL_PRIMARY_DIR}"; do
  [[ "${OUTPUT_ROOT}" != "${input_dir}" && "${OUTPUT_ROOT}" != "${input_dir}"/* ]] || { echo "output must not overwrite an input: ${OUTPUT_ROOT}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
TEMP_JOB="${JOB_SCRIPT}.partial.$$"
trap 'rm -f "${TEMP_JOB}"' EXIT
PARTITION_LINE=""
if [[ -n "${PARTITION}" ]]; then safe_value PARTITION "${PARTITION}"; PARTITION_LINE="#SBATCH --partition=${PARTITION}"; fi
cat > "${TEMP_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/ablation_%j.out
#SBATCH --error=${LOG_DIR}/ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96gb
#SBATCH --time=${TIME_LIMIT}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
${PARTITION_LINE}
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void
unset NICHERUNNER_MLP_UNIT NICHERUNNER_MLP_MODE NICHERUNNER_MLP_BACKEND NICHERUNNER_MLP_GRID_PROFILE NICHERUNNER_MLP_SELECTION_METRIC NICHERUNNER_MLP_RESAMPLING NICHERUNNER_MLP_DECISION_THRESHOLD NICHERUNNER_MLP_MAX_EPOCHS NICHERUNNER_MLP_PATIENCE NICHERUNNER_TOP_ENRICHMENT_FEATURES NICHERUNNER_TOP_NICHE_GENE_FEATURES NICHERUNNER_SKIP_SHAP NICHERUNNER_MLP_DEVICE NICHERUNNER_COMPOSITION_PREFIX NICHERUNNER_MLP_FIXED_PARAMS_PATH NICHERUNNER_MLP_BEST_PARAMS_OUT NICHERUNNER_OUTPUT_DIR NICHERUNNER_SOURCE_OUTPUT_DIR NICHERUNNER_MLP_OUTPUT_DIR
[[ -d ${REPO_DIR@Q} && -d ${NMF_FEATURE_DIR@Q} && -d ${NMF_SOURCE_DIR@Q} && -d ${NOVAE_FEATURE_DIR@Q} && -d ${FULL_PRIMARY_DIR@Q} ]]
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate ${CONDA_ENV@Q}
set -u
cd ${REPO_DIR@Q}
python scripts/run_novae_classifier_ablation.py \\
  --nmf-feature-dir ${NMF_FEATURE_DIR@Q} \\
  --nmf-source-dir ${NMF_SOURCE_DIR@Q} \\
  --novae-feature-dir ${NOVAE_FEATURE_DIR@Q} \\
  --full-primary-dir ${FULL_PRIMARY_DIR@Q} \\
  --output-root ${OUTPUT_ROOT@Q}
EOF
chmod 700 "${TEMP_JOB}"
ln "${TEMP_JOB}" "${JOB_SCRIPT}"
rm -f "${TEMP_JOB}"
trap - EXIT
if (( RENDER_ONLY )); then echo "Rendered ${JOB_SCRIPT}"; exit 0; fi
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')
echo "Queued ${JOB_ID}; output will be ${OUTPUT_ROOT}"
