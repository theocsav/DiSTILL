#!/usr/bin/env bash
# Render or submit the read-only NOVAE H5AD QC audit on UF HiPerGator.
# This launcher never opens an H5AD; real data processing occurs in SLURM only.
set -euo pipefail

RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then
  RENDER_ONLY=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--render-only|--no-submit]" >&2
  exit 2
fi

REPO_DIR="${NOVAE_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
PROCESSED_DIR="${NOVAE_PROCESSED_DIR:-/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed}"
SOURCE_H5AD="${NOVAE_SOURCE_H5AD:-${PROCESSED_DIR}/skin_visium_ssc_spatial.h5ad}"
SAMPLE_MANIFEST="${NOVAE_SAMPLE_MANIFEST:-${PROCESSED_DIR}/skin_visium_ssc_sample_manifest.csv}"
ANNOTATED_H5AD="${NOVAE_ANNOTATED_H5AD:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot/h5ad-provenance-fix-20260906_024602/novae_skin_visium_ssc_zero_shot.h5ad}"
RUN_ROOT="${NOVAE_RUN_ROOT:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot}"
OUTPUT_DIR="${NOVAE_H5AD_QC_OUTPUT_DIR:-${RUN_ROOT}/h5ad-qc-skin_visium_ssc}"
LOG_DIR="${NOVAE_H5AD_QC_LOG_DIR:-${RUN_ROOT}/h5ad-qc-logs}"
JOB_SCRIPT="${NOVAE_H5AD_QC_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_h5ad_qc.sbatch}"
CONDA_ENV="${NOVAE_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"
QOS="${NOVAE_QOS:-kejun.huang}"
PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_H5AD_QC_TIME:-01:00:00}"
DOMAIN_COLUMNS="${NOVAE_H5AD_QC_DOMAIN_COLUMNS:-}"
SLIDE_KEY="${NOVAE_H5AD_QC_SLIDE_KEY:-sample_id}"

validate_directive() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* || ! "${value}" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
    echo "${name} contains unsafe scheduler characters" >&2
    exit 2
  fi
}
validate_directive NOVAE_ACCOUNT "${ACCOUNT}"
validate_directive NOVAE_QOS "${QOS}"
validate_directive NOVAE_H5AD_QC_TIME "${TIME_LIMIT}"
if ! [[ "${TIME_LIMIT}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "NOVAE_H5AD_QC_TIME must use HH:MM:SS" >&2
  exit 2
fi
validate_directive NOVAE_H5AD_QC_SLIDE_KEY "${SLIDE_KEY}"
if [[ -n "${PARTITION}" ]]; then
  validate_directive NOVAE_PARTITION "${PARTITION}"
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
else
  PARTITION_DIRECTIVE=""
fi
validate_path() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* || ! "${value}" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
    echo "${name} contains unsafe path characters" >&2
    exit 2
  fi
}
for path_pair in \
  "NOVAE_REPO_DIR:${REPO_DIR}" "NOVAE_SOURCE_H5AD:${SOURCE_H5AD}" \
  "NOVAE_SAMPLE_MANIFEST:${SAMPLE_MANIFEST}" "NOVAE_ANNOTATED_H5AD:${ANNOTATED_H5AD}" "NOVAE_RUN_ROOT:${RUN_ROOT}" \
  "NOVAE_H5AD_QC_OUTPUT_DIR:${OUTPUT_DIR}" "NOVAE_H5AD_QC_LOG_DIR:${LOG_DIR}" \
  "NOVAE_H5AD_QC_JOB_SCRIPT:${JOB_SCRIPT}" "NOVAE_CONDA_ENV:${CONDA_ENV}"; do
  validate_path "${path_pair%%:*}" "${path_pair#*:}"
done
if [[ "${OUTPUT_DIR}" == "${SOURCE_H5AD}" || "${OUTPUT_DIR}" == "${ANNOTATED_H5AD}" ]]; then
  echo "output directory must not be an input H5AD path" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
if [[ -n "${DOMAIN_COLUMNS}" ]]; then
  if [[ "${DOMAIN_COLUMNS}" == *$'\n'* || "${DOMAIN_COLUMNS}" == *$'\r'* || ! "${DOMAIN_COLUMNS}" =~ ^[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*$ ]]; then
    echo "NOVAE_H5AD_QC_DOMAIN_COLUMNS contains unsafe characters" >&2
    exit 2
  fi
fi

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=novae_h5ad_qc
#SBATCH --output=${LOG_DIR}/novae_h5ad_qc_%j.out
#SBATCH --error=${LOG_DIR}/novae_h5ad_qc_%j.err
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
SOURCE_H5AD=${SOURCE_H5AD@Q}
SAMPLE_MANIFEST=${SAMPLE_MANIFEST@Q}
ANNOTATED_H5AD=${ANNOTATED_H5AD@Q}
OUTPUT_DIR=${OUTPUT_DIR@Q}
CONDA_ENV=${CONDA_ENV@Q}
SLIDE_KEY=${SLIDE_KEY@Q}
DOMAIN_COLUMNS=${DOMAIN_COLUMNS@Q}

module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "\${CONDA_ENV}"
set -u
cd "\${REPO_DIR}"
[[ -f "\${SOURCE_H5AD}" && -f "\${ANNOTATED_H5AD}" && -f "\${SAMPLE_MANIFEST}" ]]
DOMAIN_ARGS=()
if [[ -n "\${DOMAIN_COLUMNS}" ]]; then
  DOMAIN_ARGS=(--domain-columns "\${DOMAIN_COLUMNS}")
fi
python scripts/audit_novae_h5ad_qc.py \\
  --source-h5ad "\${SOURCE_H5AD}" \\
  --annotated-h5ad "\${ANNOTATED_H5AD}" \\
  --sample-manifest "\${SAMPLE_MANIFEST}" \\
  --output-dir ${OUTPUT_DIR@Q} \\
  --slide-key "\${SLIDE_KEY}" \\
  "\${DOMAIN_ARGS[@]}"
EOF
chmod +x "${JOB_SCRIPT}"
if (( RENDER_ONLY )); then
  echo "Rendered sbatch script: ${JOB_SCRIPT}"
  exit 0
fi
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')
echo "Queued NOVAE H5AD QC audit: ${JOB_ID}"
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  sacct -j ${JOB_ID} --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS"
echo "  tail -f ${LOG_DIR}/novae_h5ad_qc_${JOB_ID}.out"
echo "Outputs: ${OUTPUT_DIR}"
