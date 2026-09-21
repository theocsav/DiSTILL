#!/usr/bin/env bash
# CPU-only 64 GB SLURM launcher for read-only NOVAE run comparison.
set -euo pipefail
RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then RENDER_ONLY=1; shift; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--render-only|--no-submit]" >&2; exit 2; fi

REPO_DIR="${NOVAE_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
BASELINE_H5AD="${NOVAE_BASELINE_H5AD:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot/h5ad-provenance-fix-20260906_024602/novae_skin_visium_ssc_zero_shot.h5ad}"
BASELINE_MANIFEST="${NOVAE_BASELINE_MANIFEST:-${BASELINE_H5AD%/*}/novae_resolved_manifest_skin_visium_ssc.json}"
SENSITIVITY_H5AD="${NOVAE_SENSITIVITY_H5AD:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot/skin_visium_ssc_nominal_100um_sensitivity/novae_skin_visium_ssc_nominal_100um_sensitivity_zero_shot.h5ad}"
SENSITIVITY_MANIFEST="${NOVAE_SENSITIVITY_MANIFEST:-${SENSITIVITY_H5AD%/*}/novae_resolved_manifest_skin_visium_ssc_nominal_100um_sensitivity.json}"
OUTPUT_DIR="${NOVAE_COMPARISON_OUTPUT_DIR:-${REPO_DIR}/runs/novae_skin_pilot/nominal_100um_comparison}"
RUN_ROOT="${NOVAE_COMPARISON_RUN_ROOT:-${REPO_DIR}/runs/novae_skin_pilot/nominal_100um_comparison_submit}"
LOG_DIR="${NOVAE_COMPARISON_LOG_DIR:-${RUN_ROOT}/logs}"
JOB_SCRIPT="${NOVAE_COMPARISON_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_comparison.sbatch}"
CONDA_ENV="${NOVAE_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"; QOS="${NOVAE_QOS:-kejun.huang}"; PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_COMPARISON_TIME:-02:00:00}"; SLIDE_KEY="${NOVAE_COMPARISON_SLIDE_KEY:-sample_id}"

validate() { local name="$1" value="$2"; if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* || ! "$value" =~ ^[A-Za-z0-9._:/-]+$ ]]; then echo "$name contains unsafe characters" >&2; exit 2; fi; }
validate NOVAE_ACCOUNT "$ACCOUNT"; validate NOVAE_QOS "$QOS"; validate NOVAE_COMPARISON_TIME "$TIME_LIMIT"; validate NOVAE_COMPARISON_SLIDE_KEY "$SLIDE_KEY"
if ! [[ "$TIME_LIMIT" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then echo "NOVAE_COMPARISON_TIME must use HH:MM:SS" >&2; exit 2; fi
for path_pair in \
  "NOVAE_REPO_DIR:${REPO_DIR}" "NOVAE_BASELINE_H5AD:${BASELINE_H5AD}" \
  "NOVAE_BASELINE_MANIFEST:${BASELINE_MANIFEST}" "NOVAE_SENSITIVITY_H5AD:${SENSITIVITY_H5AD}" \
  "NOVAE_SENSITIVITY_MANIFEST:${SENSITIVITY_MANIFEST}" "NOVAE_COMPARISON_OUTPUT_DIR:${OUTPUT_DIR}" \
  "NOVAE_COMPARISON_RUN_ROOT:${RUN_ROOT}" "NOVAE_COMPARISON_LOG_DIR:${LOG_DIR}" \
  "NOVAE_COMPARISON_JOB_SCRIPT:${JOB_SCRIPT}" "NOVAE_CONDA_ENV:${CONDA_ENV}"; do
  path_name="${path_pair%%:*}"; path_value="${path_pair#*:}"
  if [[ -z "$path_value" || "$path_value" == *$'\n'* || "$path_value" == *$'\r'* || ! "$path_value" =~ ^[A-Za-z0-9._:/-]+$ ]]; then echo "$path_name contains unsafe path characters" >&2; exit 2; fi
done
mkdir -p "$RUN_ROOT" "$LOG_DIR"
PARTITION_DIRECTIVE=""; if [[ -n "$PARTITION" ]]; then validate NOVAE_PARTITION "$PARTITION"; PARTITION_DIRECTIVE="#SBATCH --partition=$PARTITION"; fi
cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=novae_compare
#SBATCH --output=${LOG_DIR}/novae_compare_%j.out
#SBATCH --error=${LOG_DIR}/novae_compare_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --time=${TIME_LIMIT}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
${PARTITION_DIRECTIVE}
set -euo pipefail
REPO_DIR=${REPO_DIR@Q}; BASELINE_H5AD=${BASELINE_H5AD@Q}; BASELINE_MANIFEST=${BASELINE_MANIFEST@Q}
SENSITIVITY_H5AD=${SENSITIVITY_H5AD@Q}; SENSITIVITY_MANIFEST=${SENSITIVITY_MANIFEST@Q}; OUTPUT_DIR=${OUTPUT_DIR@Q}; CONDA_ENV=${CONDA_ENV@Q}
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "\${CONDA_ENV}"; set -u
cd "\${REPO_DIR}"
[[ -f "\${BASELINE_H5AD}" && -f "\${BASELINE_MANIFEST}" && -f "\${SENSITIVITY_H5AD}" && -f "\${SENSITIVITY_MANIFEST}" ]]
python scripts/compare_novae_runs.py \\
  --baseline-h5ad "\${BASELINE_H5AD}" --baseline-manifest "\${BASELINE_MANIFEST}" \\
  --sensitivity-h5ad "\${SENSITIVITY_H5AD}" --sensitivity-manifest "\${SENSITIVITY_MANIFEST}" \\
  --output-dir "\${OUTPUT_DIR}" --slide-key ${SLIDE_KEY@Q}
EOF
chmod +x "$JOB_SCRIPT"
if (( RENDER_ONLY )); then echo "Rendered sbatch script: $JOB_SCRIPT"; exit 0; fi
JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
echo "Queued NOVAE comparison: $JOB_ID"
echo "Outputs: $OUTPUT_DIR"
