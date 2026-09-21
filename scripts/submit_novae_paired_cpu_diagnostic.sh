#!/usr/bin/env bash
# Render/submit the predeclared paired CPU calibration diagnostic.
# Both arms run sequentially in one 1-CPU, 96 GB node allocation.
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
INPUT_H5AD="${NOVAE_PAIRED_INPUT_H5AD:-/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_spatial.h5ad}"
ORIGINAL_MANIFEST="${NOVAE_PAIRED_ORIGINAL_MANIFEST:-/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_sample_manifest.csv}"
SCALE_MANIFEST="${NOVAE_PAIRED_SCALE_MANIFEST:-${REPO_DIR}/presets/novae_nominal_100um_scales.csv}"
MODEL_PATH="${NOVAE_PAIRED_MODEL_PATH:-/blue/kejun.huang/vasco.hinostroza/models/novae-human-0}"
MODEL_REVISION="b8c0a5d7612bac6bc719ab57ed3cd16ad814728c"
ORIGINAL_DATASET_ID="skin_visium_ssc_paired_cpu_original"
CALIBRATED_DATASET_ID="skin_visium_ssc_paired_cpu_calibrated"
RUN_ROOT="${NOVAE_PAIRED_RUN_ROOT:-${REPO_DIR}/runs/novae_skin_pilot/paired_cpu_diagnostic}"
ORIGINAL_OUTPUT="${NOVAE_PAIRED_ORIGINAL_OUTPUT_DIR:-${RUN_ROOT}/${ORIGINAL_DATASET_ID}}"
CALIBRATED_OUTPUT="${NOVAE_PAIRED_CALIBRATED_OUTPUT_DIR:-${RUN_ROOT}/${CALIBRATED_DATASET_ID}}"
COMPARISON_OUTPUT="${NOVAE_PAIRED_COMPARISON_OUTPUT_DIR:-${RUN_ROOT}/paired_cpu_comparison}"
LOG_DIR="${NOVAE_PAIRED_LOG_DIR:-${RUN_ROOT}/logs}"
JOB_SCRIPT="${NOVAE_PAIRED_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_paired_cpu_diagnostic.sbatch}"
CONDA_ENV="${NOVAE_PAIRED_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"
QOS="${NOVAE_QOS:-kejun.huang}"
PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_PAIRED_TIME:-24:00:00}"
JOB_NAME="${NOVAE_PAIRED_JOB_NAME:-novae_paired_cpu}"

reject_conflict() {
  local name="$1" expected="$2" value="${!1-}"
  if [[ -n "${value}" && "${value}" != "${expected}" ]]; then
    echo "${name} conflicts with the fixed paired CPU diagnostic protocol" >&2
    exit 2
  fi
}
# Scientific and execution settings are immutable. Test paths have dedicated
# NOVAE_PAIRED_* overrides above so inherited launcher settings cannot leak in.
reject_conflict NOVAE_RESOLUTIONS "0.5 1.0 2.0"
reject_conflict NOVAE_PRIMARY_RESOLUTION "1.0"
reject_conflict NOVAE_EXPECTED_NEIGHBOR_DISTANCE_UM "100"
reject_conflict NOVAE_NEIGHBOR_DISTANCE_RELATIVE_TOLERANCE "0.5"
reject_conflict NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE "0.70"
reject_conflict NOVAE_SEED "42"
reject_conflict NOVAE_WORKERS "0"
reject_conflict NOVAE_CPUS_PER_TASK "1"
reject_conflict NOVAE_ACCELERATOR "cpu"
reject_conflict NOVAE_DETERMINISTIC "1"
reject_conflict NOVAE_COORDINATE_STRATEGY "visium_manifest"
reject_conflict NOVAE_OMIT_GRAPH_RADIUS_PRUNING "0"
reject_conflict NOVAE_DATASET_ID "${ORIGINAL_DATASET_ID}"
reject_conflict NOVAE_INPUT_H5AD "${INPUT_H5AD}"
reject_conflict NOVAE_SAMPLE_MANIFEST "${ORIGINAL_MANIFEST}"
reject_conflict NOVAE_MODEL_PATH "${MODEL_PATH}"
reject_conflict NOVAE_MODEL_SOURCE "${MODEL_PATH}"
for inherited_name in NOVAE_OUTPUT_DIR NOVAE_RUN_ROOT NOVAE_LOG_DIR NOVAE_JOB_SCRIPT; do
  if [[ -n "${!inherited_name-}" ]]; then
    echo "${inherited_name} conflicts with the fixed paired CPU diagnostic; use NOVAE_PAIRED_*" >&2
    exit 2
  fi
done
if [[ -n "${NOVAE_MODEL_REVISION-}" && "${NOVAE_MODEL_REVISION}" != "${MODEL_REVISION}" ]]; then
  echo "NOVAE_MODEL_REVISION conflicts with the fixed paired CPU diagnostic revision" >&2
  exit 2
fi

validate_path() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* || ! "${value}" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
    echo "${name} contains unsafe path characters" >&2
    exit 2
  fi
}
validate_directive() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* || ! "${value}" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
    echo "${name} contains unsafe scheduler characters" >&2
    exit 2
  fi
}
for pair in \
  "NOVAE_REPO_DIR:${REPO_DIR}" "NOVAE_PAIRED_INPUT_H5AD:${INPUT_H5AD}" \
  "NOVAE_PAIRED_ORIGINAL_MANIFEST:${ORIGINAL_MANIFEST}" "NOVAE_PAIRED_SCALE_MANIFEST:${SCALE_MANIFEST}" \
  "NOVAE_PAIRED_MODEL_PATH:${MODEL_PATH}" "NOVAE_PAIRED_RUN_ROOT:${RUN_ROOT}" \
  "NOVAE_PAIRED_ORIGINAL_OUTPUT_DIR:${ORIGINAL_OUTPUT}" "NOVAE_PAIRED_CALIBRATED_OUTPUT_DIR:${CALIBRATED_OUTPUT}" \
  "NOVAE_PAIRED_COMPARISON_OUTPUT_DIR:${COMPARISON_OUTPUT}" "NOVAE_PAIRED_LOG_DIR:${LOG_DIR}" \
  "NOVAE_PAIRED_JOB_SCRIPT:${JOB_SCRIPT}" "NOVAE_PAIRED_CONDA_ENV:${CONDA_ENV}"; do
  validate_path "${pair%%:*}" "${pair#*:}"
done
validate_directive NOVAE_ACCOUNT "${ACCOUNT}"
validate_directive NOVAE_QOS "${QOS}"
validate_directive NOVAE_PAIRED_TIME "${TIME_LIMIT}"
validate_directive NOVAE_PAIRED_JOB_NAME "${JOB_NAME}"
if ! [[ "${TIME_LIMIT}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "NOVAE_PAIRED_TIME must use HH:MM:SS" >&2
  exit 2
fi
if [[ "${ORIGINAL_OUTPUT}" == "${CALIBRATED_OUTPUT}" || "${ORIGINAL_OUTPUT}" == "${COMPARISON_OUTPUT}" || "${CALIBRATED_OUTPUT}" == "${COMPARISON_OUTPUT}" ]]; then
  echo "paired output identities must be distinct" >&2
  exit 2
fi
if [[ "${ORIGINAL_MANIFEST}" == "${SCALE_MANIFEST}" ]]; then
  echo "original and calibrated manifests must be distinct immutable inputs" >&2
  exit 2
fi
if [[ "${ORIGINAL_DATASET_ID}" == "${CALIBRATED_DATASET_ID}" ]]; then
  echo "paired dataset identities must be distinct" >&2
  exit 2
fi
for output in "${ORIGINAL_OUTPUT}" "${CALIBRATED_OUTPUT}" "${COMPARISON_OUTPUT}"; do
  if [[ -e "${output}" ]]; then
    echo "refusing existing paired output path: ${output}" >&2
    exit 2
  fi
done
if [[ -n "${PARTITION}" ]]; then
  validate_directive NOVAE_PARTITION "${PARTITION}"
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
else
  PARTITION_DIRECTIVE=""
fi
if [[ -e "${JOB_SCRIPT}" ]]; then
  echo "refusing to overwrite existing job script: ${JOB_SCRIPT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
TEMP_JOB="${JOB_SCRIPT}.partial.$$"
trap 'rm -f "${TEMP_JOB}"' EXIT
cat > "${TEMP_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/novae_paired_cpu_%j.out
#SBATCH --error=${LOG_DIR}/novae_paired_cpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=96gb
#SBATCH --time=${TIME_LIMIT}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
${PARTITION_DIRECTIVE}
set -euo pipefail

REPO_DIR=${REPO_DIR@Q}
INPUT_H5AD=${INPUT_H5AD@Q}
ORIGINAL_MANIFEST=${ORIGINAL_MANIFEST@Q}
SCALE_MANIFEST=${SCALE_MANIFEST@Q}
MODEL_PATH=${MODEL_PATH@Q}
MODEL_REVISION=${MODEL_REVISION@Q}
CONDA_ENV=${CONDA_ENV@Q}
ORIGINAL_OUTPUT=${ORIGINAL_OUTPUT@Q}
CALIBRATED_OUTPUT=${CALIBRATED_OUTPUT@Q}
COMPARISON_OUTPUT=${COMPARISON_OUTPUT@Q}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

[[ -d "\${REPO_DIR}" && -f "\${INPUT_H5AD}" && -f "\${ORIGINAL_MANIFEST}" && -f "\${SCALE_MANIFEST}" && -d "\${MODEL_PATH}" ]]
[[ ! -e "\${ORIGINAL_OUTPUT}" && ! -e "\${CALIBRATED_OUTPUT}" && ! -e "\${COMPARISON_OUTPUT}" ]]
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "\${CONDA_ENV}"
set -u
cd "\${REPO_DIR}"

# The two invocations are intentionally sequential and share this node's
# environment; run_novae_pilot publishes each output transactionally.
python scripts/run_novae_pilot.py \\
  --input-h5ad "\${INPUT_H5AD}" --output-dir "\${ORIGINAL_OUTPUT}" \\
  --dataset-id ${ORIGINAL_DATASET_ID@Q} --slide-key sample_id --group-key patient \\
  --technology visium --expression-mode raw_counts --model-source "\${MODEL_PATH}" \\
  --model-revision "\${MODEL_REVISION}" --resolutions 0.5 1.0 2.0 --primary-resolution 1.0 \\
  --expected-neighbor-distance-um 100 --neighbor-distance-relative-tolerance 0.5 \\
  --min-domain-assignment-coverage 0.70 --accelerator cpu --workers 0 --seed 42 --deterministic \\
  --coordinate-strategy visium_manifest --sample-manifest "\${ORIGINAL_MANIFEST}" \\
  --physical-spot-diameter-um 55.0 --graph-radius-um 100

python scripts/run_novae_pilot.py \\
  --input-h5ad "\${INPUT_H5AD}" --output-dir "\${CALIBRATED_OUTPUT}" \\
  --dataset-id ${CALIBRATED_DATASET_ID@Q} --slide-key sample_id --group-key patient \\
  --technology visium --expression-mode raw_counts --model-source "\${MODEL_PATH}" \\
  --model-revision "\${MODEL_REVISION}" --resolutions 0.5 1.0 2.0 --primary-resolution 1.0 \\
  --expected-neighbor-distance-um 100 --neighbor-distance-relative-tolerance 0.5 \\
  --min-domain-assignment-coverage 0.70 --accelerator cpu --workers 0 --seed 42 --deterministic \\
  --coordinate-strategy visium_explicit_scale --sample-manifest "\${SCALE_MANIFEST}"

python scripts/compare_novae_runs.py \\
  --baseline-h5ad "\${ORIGINAL_OUTPUT}/novae_${ORIGINAL_DATASET_ID}_zero_shot.h5ad" \\
  --baseline-manifest "\${ORIGINAL_OUTPUT}/novae_resolved_manifest_${ORIGINAL_DATASET_ID}.json" \\
  --sensitivity-h5ad "\${CALIBRATED_OUTPUT}/novae_${CALIBRATED_DATASET_ID}_zero_shot.h5ad" \\
  --sensitivity-manifest "\${CALIBRATED_OUTPUT}/novae_resolved_manifest_${CALIBRATED_DATASET_ID}.json" \\
  --output-dir "\${COMPARISON_OUTPUT}" --slide-key sample_id
EOF
chmod 700 "${TEMP_JOB}"
if ! ln "${TEMP_JOB}" "${JOB_SCRIPT}"; then
  echo "refusing to overwrite existing job script: ${JOB_SCRIPT}" >&2
  exit 2
fi
rm -f "${TEMP_JOB}"
trap - EXIT
if (( RENDER_ONLY )); then
  echo "Rendered paired CPU diagnostic: ${JOB_SCRIPT}"
  exit 0
fi
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')
echo "Queued paired NOVAE CPU diagnostic: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Outputs: ${ORIGINAL_OUTPUT}, ${CALIBRATED_OUTPUT}, ${COMPARISON_OUTPUT}"
