#!/usr/bin/env bash
# Render or submit the exact patient-level McNemar report from immutable pooling output.
set -euo pipefail

RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then RENDER_ONLY=1; shift; fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--render-only|--no-submit]" >&2; exit 2; }

REPO_DIR="${NOVAE_PATIENT_LEVEL_MCNEMAR_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
INPUT_ROOT="${NOVAE_PATIENT_LEVEL_MCNEMAR_INPUT_ROOT:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_patient_level_pooling_20260924T161344Z_3258476/historical164_patient_level_pooling}"
RUN_ROOT="${NOVAE_PATIENT_LEVEL_MCNEMAR_RUN_ROOT:-${REPO_DIR}/runs/novae_patient_level_mcnemar_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUTPUT_ROOT="${NOVAE_PATIENT_LEVEL_MCNEMAR_OUTPUT_ROOT:-${RUN_ROOT}/historical164_patient_level_mcnemar}"
JOB_SCRIPT="${NOVAE_PATIENT_LEVEL_MCNEMAR_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_patient_level_mcnemar.sbatch}"
LOG_DIR="${NOVAE_PATIENT_LEVEL_MCNEMAR_LOG_DIR:-${RUN_ROOT}/logs}"
CONDA_ENV="${NOVAE_PATIENT_LEVEL_MCNEMAR_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"
QOS="${NOVAE_QOS:-kejun.huang}"
PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_PATIENT_LEVEL_MCNEMAR_TIME:-01:00:00}"
JOB_NAME="${NOVAE_PATIENT_LEVEL_MCNEMAR_JOB_NAME:-novae_mcnemar}"

safe_path() { [[ -n "$2" && "$2" != *$'\n'* && "$2" != *$'\r'* && "$2" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "$1 contains unsafe path characters" >&2; exit 2; }; }
safe_value() { [[ -n "$2" && "$2" != *$'\n'* && "$2" != *$'\r'* && "$2" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "$1 contains unsafe scheduler characters" >&2; exit 2; }; }
for pair in "REPO_DIR:${REPO_DIR}" "INPUT_ROOT:${INPUT_ROOT}" "RUN_ROOT:${RUN_ROOT}" "OUTPUT_ROOT:${OUTPUT_ROOT}" "JOB_SCRIPT:${JOB_SCRIPT}" "LOG_DIR:${LOG_DIR}" "CONDA_ENV:${CONDA_ENV}"; do safe_path "${pair%%:*}" "${pair#*:}"; done
for pair in "ACCOUNT:${ACCOUNT}" "QOS:${QOS}" "TIME_LIMIT:${TIME_LIMIT}" "JOB_NAME:${JOB_NAME}"; do safe_value "${pair%%:*}" "${pair#*:}"; done
[[ "${TIME_LIMIT}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "TIME_LIMIT must be HH:MM:SS" >&2; exit 2; }
[[ -d "${REPO_DIR}" && -f "${REPO_DIR}/scripts/run_novae_patient_level_mcnemar.py" ]] || { echo "repository/report script not found" >&2; exit 2; }
# Input existence is checked inside the generated job so --render-only works without
# local access to the immutable HPG output.
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing existing output root: ${OUTPUT_ROOT}" >&2; exit 2; }
[[ ! -e "${JOB_SCRIPT}" ]] || { echo "refusing existing job script: ${JOB_SCRIPT}" >&2; exit 2; }
[[ "${OUTPUT_ROOT}" != "${INPUT_ROOT}" && "${OUTPUT_ROOT}" != "${INPUT_ROOT}"/* && "${INPUT_ROOT}" != "${OUTPUT_ROOT}"/* ]] || { echo "output must not overlap immutable input" >&2; exit 2; }
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
TEMP_JOB="${JOB_SCRIPT}.partial.$$"
trap 'rm -f "${TEMP_JOB}"' EXIT
PARTITION_LINE=""
if [[ -n "${PARTITION}" ]]; then safe_value PARTITION "${PARTITION}"; PARTITION_LINE="#SBATCH --partition=${PARTITION}"; fi
cat > "${TEMP_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/mcnemar_%j.out
#SBATCH --error=${LOG_DIR}/mcnemar_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
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
[[ -d ${REPO_DIR@Q} && -d ${INPUT_ROOT@Q} ]]
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate ${CONDA_ENV@Q}
set -u
cd ${REPO_DIR@Q}
python scripts/run_novae_patient_level_mcnemar.py \\
  --input-root ${INPUT_ROOT@Q} \\
  --output-root ${OUTPUT_ROOT@Q}
EOF
chmod 700 "${TEMP_JOB}"
ln "${TEMP_JOB}" "${JOB_SCRIPT}"
rm -f "${TEMP_JOB}"
trap - EXIT
if (( RENDER_ONLY )); then echo "Rendered ${JOB_SCRIPT}"; exit 0; fi
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')
echo "Queued ${JOB_ID}; output will be ${OUTPUT_ROOT}"
