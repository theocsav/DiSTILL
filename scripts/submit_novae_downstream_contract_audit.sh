#!/usr/bin/env bash
# Render or submit the read-only skin NOVAE downstream contract audit.
# This launcher never opens data; the generated CPU job does all H5AD/table reads.
set -euo pipefail

RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then
  RENDER_ONLY=1
  shift
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--render-only|--no-submit]" >&2; exit 2; }

REPO_DIR="${NOVAE_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
HISTORICAL_SOURCE="${NOVAE_CONTRACT_HISTORICAL_SOURCE:-/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_1mmfov_spatial.h5ad}"
HISTORICAL_RUN="${NOVAE_CONTRACT_HISTORICAL_RUN_OUTPUT:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs}"
FULLSWEEP_SOURCE="${NOVAE_CONTRACT_FULLSWEEP_SOURCE:-/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_1000umfov_spatial.h5ad}"
FULLSWEEP_RUN="${NOVAE_CONTRACT_FULLSWEEP_RUN_OUTPUT:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1000umfov_poisson75_fullsweep/outputs}"
RUN_ROOT="${NOVAE_CONTRACT_AUDIT_RUN_ROOT:-${REPO_DIR}/runs/novae_downstream_contract_audit}"
OUTPUT_DIR="${NOVAE_CONTRACT_AUDIT_OUTPUT_DIR:-${RUN_ROOT}/audit}"
LOG_DIR="${NOVAE_CONTRACT_AUDIT_LOG_DIR:-${RUN_ROOT}/logs}"
JOB_SCRIPT="${NOVAE_CONTRACT_AUDIT_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_downstream_contract_audit.sbatch}"
CONDA_ENV="${NOVAE_CONTRACT_AUDIT_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"
QOS="${NOVAE_QOS:-kejun.huang}"
PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_CONTRACT_AUDIT_TIME:-02:00:00}"

validate_value() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* || ! "${value}" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
    echo "${name} contains unsafe scheduler/path characters" >&2
    exit 2
  fi
}
for pair in \
  "NOVAE_REPO_DIR:${REPO_DIR}" "NOVAE_CONTRACT_HISTORICAL_SOURCE:${HISTORICAL_SOURCE}" \
  "NOVAE_CONTRACT_HISTORICAL_RUN_OUTPUT:${HISTORICAL_RUN}" "NOVAE_CONTRACT_FULLSWEEP_SOURCE:${FULLSWEEP_SOURCE}" \
  "NOVAE_CONTRACT_FULLSWEEP_RUN_OUTPUT:${FULLSWEEP_RUN}" "NOVAE_CONTRACT_AUDIT_RUN_ROOT:${RUN_ROOT}" \
  "NOVAE_CONTRACT_AUDIT_OUTPUT_DIR:${OUTPUT_DIR}" "NOVAE_CONTRACT_AUDIT_LOG_DIR:${LOG_DIR}" \
  "NOVAE_CONTRACT_AUDIT_JOB_SCRIPT:${JOB_SCRIPT}" "NOVAE_CONTRACT_AUDIT_CONDA_ENV:${CONDA_ENV}"; do
  validate_value "${pair%%:*}" "${pair#*:}"
done
validate_value NOVAE_ACCOUNT "${ACCOUNT}"
validate_value NOVAE_QOS "${QOS}"
validate_value NOVAE_CONTRACT_AUDIT_TIME "${TIME_LIMIT}"
[[ "${TIME_LIMIT}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "NOVAE_CONTRACT_AUDIT_TIME must use HH:MM:SS" >&2; exit 2; }
if [[ -n "${PARTITION}" ]]; then
  validate_value NOVAE_PARTITION "${PARTITION}"
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
else
  PARTITION_DIRECTIVE=""
fi

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=novae_contract_audit
#SBATCH --output=${LOG_DIR}/novae_contract_audit_%j.out
#SBATCH --error=${LOG_DIR}/novae_contract_audit_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --time=${TIME_LIMIT}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
${PARTITION_DIRECTIVE}
set -euo pipefail

REPO_DIR=${REPO_DIR@Q}
HISTORICAL_SOURCE=${HISTORICAL_SOURCE@Q}
HISTORICAL_RUN=${HISTORICAL_RUN@Q}
FULLSWEEP_SOURCE=${FULLSWEEP_SOURCE@Q}
FULLSWEEP_RUN=${FULLSWEEP_RUN@Q}
OUTPUT_DIR=${OUTPUT_DIR@Q}
CONDA_ENV=${CONDA_ENV@Q}

module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "\${CONDA_ENV}"
set -u
cd "\${REPO_DIR}"
[[ ! -e "\${OUTPUT_DIR}" ]] || { echo "refusing existing output directory: \${OUTPUT_DIR}" >&2; exit 2; }
python scripts/audit_novae_downstream_contract.py \\
  --historical-source "\${HISTORICAL_SOURCE}" \\
  --historical-run-output "\${HISTORICAL_RUN}" \\
  --fullsweep-source "\${FULLSWEEP_SOURCE}" \\
  --fullsweep-run-output "\${FULLSWEEP_RUN}" \\
  --output-dir "\${OUTPUT_DIR}"
EOF
chmod +x "${JOB_SCRIPT}"
if (( RENDER_ONLY )); then
  echo "Rendered sbatch script: ${JOB_SCRIPT}"
  exit 0
fi
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')
echo "Queued NOVAE downstream contract audit: ${JOB_ID}"
echo "Outputs: ${OUTPUT_DIR}"
