#!/usr/bin/env bash
# Submit the standalone exploratory NOVAE skin pilot from an HPG login node.
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
DATA_DIR="${NOVAE_DATA_DIR:-/blue/kejun.huang/vasco.hinostroza/data/skin_dataset}"
PROCESSED_DIR="${NOVAE_PROCESSED_DIR:-${DATA_DIR}/processed}"
INPUT_H5AD="${NOVAE_INPUT_H5AD:-${PROCESSED_DIR}/skin_visium_ssc_spatial.h5ad}"
SAMPLE_MANIFEST="${NOVAE_SAMPLE_MANIFEST:-${PROCESSED_DIR}/skin_visium_ssc_sample_manifest.csv}"
DATASET_ID="${NOVAE_DATASET_ID:-skin_visium_ssc}"
RUN_ROOT="${NOVAE_RUN_ROOT:-${REPO_DIR}/runs/novae_skin_pilot}"
LOG_DIR="${NOVAE_LOG_DIR:-${RUN_ROOT}/logs}"
OUTPUT_DIR="${NOVAE_OUTPUT_DIR:-${RUN_ROOT}/${DATASET_ID}}"
CONDA_ENV="${NOVAE_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312}"
MODEL_SOURCE="${NOVAE_MODEL_PATH:-prism-oncology/novae-human-0}"
# Observed current model commit; NOVAE_MODEL_REVISION remains an override.
MODEL_REVISION="${NOVAE_MODEL_REVISION:-b8c0a5d7612bac6bc719ab57ed3cd16ad814728c}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"
QOS="${NOVAE_QOS:-kejun.huang}"
PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_TIME:-24:00:00}"
RESOLUTIONS_STRING="${NOVAE_RESOLUTIONS:-0.5 1.0 2.0}"
PRIMARY_RESOLUTION="${NOVAE_PRIMARY_RESOLUTION:-1.0}"
EXPECTED_DISTANCE_UM="${NOVAE_EXPECTED_NEIGHBOR_DISTANCE_UM:-100}"
DISTANCE_TOLERANCE="${NOVAE_NEIGHBOR_DISTANCE_RELATIVE_TOLERANCE:-0.5}"
GRAPH_RADIUS_UM="${NOVAE_GRAPH_RADIUS_UM:-100}"
WORKERS="${NOVAE_WORKERS:-8}"
SEED="${NOVAE_SEED:-42}"
JOB_SCRIPT="${RUN_ROOT}/submit_novae_skin_pilot.sbatch"

if ! [[ "${DATASET_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "NOVAE_DATASET_ID is not a safe dataset slug" >&2
  exit 2
fi
validate_directive() {
  local name="$1" value="$2"
  if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* || ! "${value}" =~ ^[A-Za-z0-9._:/-]+$ ]]; then
    echo "${name} contains unsafe scheduler characters" >&2
    exit 2
  fi
}
validate_directive NOVAE_ACCOUNT "${ACCOUNT}"
validate_directive NOVAE_QOS "${QOS}"
validate_directive NOVAE_TIME "${TIME_LIMIT}"
read -r -a RESOLUTIONS <<< "${RESOLUTIONS_STRING}"
if (( ${#RESOLUTIONS[@]} == 0 )); then
  echo "NOVAE_RESOLUTIONS must contain at least one value" >&2
  exit 2
fi
for resolution in "${RESOLUTIONS[@]}"; do
  if ! [[ "${resolution}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v value="${resolution}" 'BEGIN { exit !(value > 0 && value < 1e308) }'; then
    echo "NOVAE_RESOLUTIONS must contain positive numeric values" >&2
    exit 2
  fi
done
for ((i=0; i<${#RESOLUTIONS[@]}; i++)); do
  for ((j=i+1; j<${#RESOLUTIONS[@]}; j++)); do
    if awk -v left="${RESOLUTIONS[i]}" -v right="${RESOLUTIONS[j]}" 'BEGIN { exit !(left + 0 == right + 0) }'; then
      echo "NOVAE_RESOLUTIONS must not contain duplicates" >&2
      exit 2
    fi
  done
done
if ! [[ "${PRIMARY_RESOLUTION}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v value="${PRIMARY_RESOLUTION}" 'BEGIN { exit !(value > 0 && value < 1e308) }'; then
  echo "NOVAE_PRIMARY_RESOLUTION must be a positive numeric value" >&2
  exit 2
fi
if ! printf '%s\n' "${RESOLUTIONS[@]}" | awk -v primary="${PRIMARY_RESOLUTION}" '$0 + 0 == primary + 0 {found=1} END {exit !found}'; then
  echo "NOVAE_PRIMARY_RESOLUTION must be included in NOVAE_RESOLUTIONS" >&2
  exit 2
fi
if ! awk -v value="${EXPECTED_DISTANCE_UM}" 'BEGIN { exit !(value > 0 && value < 1e308) }' || ! [[ "${EXPECTED_DISTANCE_UM}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "NOVAE_EXPECTED_NEIGHBOR_DISTANCE_UM must be a positive numeric value" >&2
  exit 2
fi
if ! awk -v value="${DISTANCE_TOLERANCE}" 'BEGIN { exit !(value >= 0 && value < 1) }' || ! [[ "${DISTANCE_TOLERANCE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "NOVAE_NEIGHBOR_DISTANCE_RELATIVE_TOLERANCE must be in [0,1)" >&2
  exit 2
fi
if ! awk -v value="${GRAPH_RADIUS_UM}" 'BEGIN { exit !(value > 0 && value < 1e308) }' || ! [[ "${GRAPH_RADIUS_UM}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "NOVAE_GRAPH_RADIUS_UM must be a positive finite numeric value" >&2
  exit 2
fi
RESOLUTION_ARGS="--resolutions ${RESOLUTIONS[*]}"
if ! [[ "${WORKERS}" =~ ^[0-9]+$ && "${SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "NOVAE_WORKERS/NOVAE_SEED must be integer values" >&2
  exit 2
fi
if [[ -n "${PARTITION}" ]]; then
  validate_directive NOVAE_PARTITION "${PARTITION}"
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
else
  PARTITION_DIRECTIVE=""
fi
for path_value in "${REPO_DIR}" "${INPUT_H5AD}" "${SAMPLE_MANIFEST}" "${RUN_ROOT}" "${LOG_DIR}" "${OUTPUT_DIR}" "${CONDA_ENV}"; do
  if [[ "${path_value}" == *$'\n'* || "${path_value}" == *$'\r'* ]]; then
    echo "path settings must not contain newlines" >&2
    exit 2
  fi
done

# Create only parents needed to submit and monitor. The pilot itself refuses an
# existing final output directory and atomically publishes its staging sibling.
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=novae_skin_pilot
#SBATCH --output=${LOG_DIR}/novae_skin_pilot_%j.out
#SBATCH --error=${LOG_DIR}/novae_skin_pilot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=96gb
#SBATCH --time=${TIME_LIMIT}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
${PARTITION_DIRECTIVE}
set -euo pipefail

REPO_DIR=${REPO_DIR@Q}
INPUT_H5AD=${INPUT_H5AD@Q}
SAMPLE_MANIFEST=${SAMPLE_MANIFEST@Q}
OUTPUT_DIR=${OUTPUT_DIR@Q}
CONDA_ENV=${CONDA_ENV@Q}
MODEL_SOURCE=${MODEL_SOURCE@Q}
MODEL_REVISION=${MODEL_REVISION@Q}

module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "\${CONDA_ENV}"
set -u
cd "\${REPO_DIR}"

MODEL_REVISION_ARGS=()
if [[ -n "\${MODEL_REVISION}" ]]; then
  MODEL_REVISION_ARGS+=(--model-revision "\${MODEL_REVISION}")
fi
python scripts/run_novae_pilot.py \\
  --input-h5ad "\${INPUT_H5AD}" \\
  --output-dir "\${OUTPUT_DIR}" \\
  --dataset-id ${DATASET_ID@Q} \\
  --slide-key sample_id \\
  --group-key patient \\
  --technology visium \\
  --expression-mode raw_counts \\
  --model-source "\${MODEL_SOURCE}" \\
  "\${MODEL_REVISION_ARGS[@]}" \\
  ${RESOLUTION_ARGS} \\
  --primary-resolution ${PRIMARY_RESOLUTION} \\
  --expected-neighbor-distance-um ${EXPECTED_DISTANCE_UM} \\
  --neighbor-distance-relative-tolerance ${DISTANCE_TOLERANCE} \\
  --graph-radius-um ${GRAPH_RADIUS_UM} \\
  --accelerator gpu \\
  --workers ${WORKERS} \\
  --seed ${SEED} \\
  --coordinate-strategy visium_manifest \\
  --sample-manifest "\${SAMPLE_MANIFEST}" \\
  --physical-spot-diameter-um 55.0
EOF
chmod +x "${JOB_SCRIPT}"
if (( RENDER_ONLY )); then
  echo "Rendered sbatch script: ${JOB_SCRIPT}"
  exit 0
fi
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')
echo "Queued NOVAE skin pilot: ${JOB_ID}"
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  sacct -j ${JOB_ID} --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS"
echo "  tail -f ${LOG_DIR}/novae_skin_pilot_${JOB_ID}.out"
echo "Outputs: ${OUTPUT_DIR}"
