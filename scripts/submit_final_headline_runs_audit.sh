#!/usr/bin/env bash
# Render or submit the read-only final/headline skin+kidney audit on HPG.
# No H5AD is opened by this launcher; real data processing is SLURM-only.
set -euo pipefail

RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then RENDER_ONLY=1; shift; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--render-only|--no-submit]" >&2; exit 2; fi

ROOT="${AUDIT_ROOT:-/blue/kejun.huang/vasco.hinostroza}"
REPO_DIR="${AUDIT_REPO_DIR:-${ROOT}/nicherunner/src/sptx-tool}"
PROCESSED="${AUDIT_PROCESSED_DIR:-${ROOT}/data/skin_dataset/processed}"
KIDNEY_PROCESSED="${AUDIT_KIDNEY_PROCESSED_DIR:-${ROOT}/data/kidney_dataset/processed}"
RUNS="${AUDIT_RUNS_DIR:-${ROOT}/nicherunner/src/sptx-tool/runs}"
SKIN_H5AD="${AUDIT_SKIN_H5AD:-${PROCESSED}/skin_visium_ssc_1mmfov_spatial.h5ad}"
KIDNEY_SPATIAL_H5AD="${AUDIT_KIDNEY_SPATIAL_H5AD:-${KIDNEY_PROCESSED}/kidney_cosmx_six_sample_spatial.h5ad}"
KIDNEY_REFERENCE_H5AD="${AUDIT_KIDNEY_REFERENCE_H5AD:-${KIDNEY_PROCESSED}/gse183277_kidney_reference.h5ad}"
SKIN_SPLIT="${AUDIT_SKIN_SPLIT_RUN_DIR:-${RUNS}/skin_visium_ssc_1mmfov_poisson75_split/outputs}"
SKIN_1000="${AUDIT_SKIN_1000_RUN_DIR:-${RUNS}/skin_visium_ssc_1000umfov_poisson75_fullsweep/outputs}"
SKIN_750="${AUDIT_SKIN_750_RUN_DIR:-${RUNS}/skin_visium_ssc_750umfov_poisson75_fullsweep/outputs}"
SKIN_500="${AUDIT_SKIN_500_RUN_DIR:-${RUNS}/skin_visium_ssc_500umfov_poisson75_fullsweep/outputs}"
KIDNEY_RUN="${AUDIT_KIDNEY_RUN_DIR:-${RUNS}/kidney_cosmx_ssc_poisson75/outputs}"
OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-${RUNS}/final_headline_runs_audit}"
LOG_DIR="${AUDIT_LOG_DIR:-${RUNS}/final_headline_runs_audit_logs}"
JOB_SCRIPT="${AUDIT_JOB_SCRIPT:-${LOG_DIR}/submit_final_headline_runs_audit.sbatch}"
CONDA_ENV="${AUDIT_CONDA_ENV:-${ROOT}/nicherunner/conda/envs/ibd_cosmx_k4}"
ACCOUNT="${AUDIT_ACCOUNT:-kejun.huang}"
QOS="${AUDIT_QOS:-kejun.huang-b}"
TIME_LIMIT="${AUDIT_TIME:-04:00:00}"
PARTITION="${AUDIT_PARTITION:-}"

validate_value() {
  local name="$1" value="$2"
  if [[ -z "$value" || "$value" == *$'\n'* || "$value" == *$'\r'* || "$value" == *[!A-Za-z0-9._:/\ -]* ]]; then
    echo "$name contains unsafe characters" >&2; exit 2
  fi
}
validate_no_whitespace() {
  local name="$1" value="$2"
  if [[ "$value" == *[[:space:]]* ]]; then
    echo "$name must not contain whitespace because it is used in #SBATCH output/error paths" >&2; exit 2
  fi
}
validate_value AUDIT_ACCOUNT "$ACCOUNT"; validate_value AUDIT_QOS "$QOS"; validate_value AUDIT_TIME "$TIME_LIMIT"
validate_no_whitespace AUDIT_ACCOUNT "$ACCOUNT"; validate_no_whitespace AUDIT_QOS "$QOS"
validate_no_whitespace AUDIT_LOG_DIR "$LOG_DIR"; validate_no_whitespace AUDIT_JOB_SCRIPT "$JOB_SCRIPT"
if [[ ! "$TIME_LIMIT" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then echo "AUDIT_TIME must use HH:MM:SS" >&2; exit 2; fi
for pair in \
  "AUDIT_REPO_DIR:$REPO_DIR" "AUDIT_CONDA_ENV:$CONDA_ENV" "AUDIT_SKIN_H5AD:$SKIN_H5AD" \
  "AUDIT_KIDNEY_SPATIAL_H5AD:$KIDNEY_SPATIAL_H5AD" "AUDIT_KIDNEY_REFERENCE_H5AD:$KIDNEY_REFERENCE_H5AD" \
  "AUDIT_SKIN_SPLIT_RUN_DIR:$SKIN_SPLIT" "AUDIT_SKIN_1000_RUN_DIR:$SKIN_1000" "AUDIT_SKIN_750_RUN_DIR:$SKIN_750" \
  "AUDIT_SKIN_500_RUN_DIR:$SKIN_500" "AUDIT_KIDNEY_RUN_DIR:$KIDNEY_RUN" "AUDIT_OUTPUT_DIR:$OUTPUT_DIR" \
  "AUDIT_LOG_DIR:$LOG_DIR" "AUDIT_JOB_SCRIPT:$JOB_SCRIPT"; do
  validate_value "${pair%%:*}" "${pair#*:}"
done
mkdir -p "$LOG_DIR"
if [[ -n "$PARTITION" ]]; then
  validate_value AUDIT_PARTITION "$PARTITION"
  validate_no_whitespace AUDIT_PARTITION "$PARTITION"
  PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"
else
  PARTITION_DIRECTIVE=""
fi

cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=final_headline_audit
#SBATCH --output=${LOG_DIR}/final_headline_audit_%j.out
#SBATCH --error=${LOG_DIR}/final_headline_audit_%j.err
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
CONDA_ENV=${CONDA_ENV@Q}
SKIN_H5AD=${SKIN_H5AD@Q}
KIDNEY_SPATIAL_H5AD=${KIDNEY_SPATIAL_H5AD@Q}
KIDNEY_REFERENCE_H5AD=${KIDNEY_REFERENCE_H5AD@Q}
SKIN_SPLIT=${SKIN_SPLIT@Q}
SKIN_1000=${SKIN_1000@Q}
SKIN_750=${SKIN_750@Q}
SKIN_500=${SKIN_500@Q}
KIDNEY_RUN=${KIDNEY_RUN@Q}
OUTPUT_DIR=${OUTPUT_DIR@Q}

module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "\${CONDA_ENV}"
set -u
cd "\${REPO_DIR}"
python scripts/audit_final_headline_runs.py \\
  --skin-h5ad "\${SKIN_H5AD}" \\
  --kidney-spatial-h5ad "\${KIDNEY_SPATIAL_H5AD}" \\
  --kidney-reference-h5ad "\${KIDNEY_REFERENCE_H5AD}" \\
  --skin-split-run-dir "\${SKIN_SPLIT}" \\
  --skin-1000-run-dir "\${SKIN_1000}" \\
  --skin-750-run-dir "\${SKIN_750}" \\
  --skin-500-run-dir "\${SKIN_500}" \\
  --kidney-run-dir "\${KIDNEY_RUN}" \\
  --output-dir "\${OUTPUT_DIR}"
EOF
chmod +x "$JOB_SCRIPT"
if (( RENDER_ONLY )); then echo "Rendered sbatch script: $JOB_SCRIPT"; exit 0; fi
JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
echo "Queued final/headline audit: $JOB_ID"
echo "Outputs: $OUTPUT_DIR"
