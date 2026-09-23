#!/usr/bin/env bash
# Render/submit the read-only NOVAE spatial biological validation.
set -euo pipefail
RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then RENDER_ONLY=1; shift; fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--render-only|--no-submit]" >&2; exit 2; }
REPO_DIR="${NOVAE_SPATIAL_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
H5AD="${NOVAE_SPATIAL_H5AD:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot/paired_cpu_diagnostic/skin_visium_ssc_paired_cpu_calibrated/novae_skin_visium_ssc_paired_cpu_calibrated_zero_shot.h5ad}"
POST="${NOVAE_SPATIAL_POST_NMF_OBS:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/post_nmf_obs.csv}"
RUN_ROOT="${NOVAE_SPATIAL_RUN_ROOT:-${REPO_DIR}/runs/novae_spatial_biological_validation_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUTPUT="${NOVAE_SPATIAL_OUTPUT_DIR:-${RUN_ROOT}/validation}"
JOB="${NOVAE_SPATIAL_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_spatial_biological_validation.sbatch}"
LOG="${NOVAE_SPATIAL_LOG_DIR:-${RUN_ROOT}/logs}"
ENV_PATH="${NOVAE_SPATIAL_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"; QOS="${NOVAE_QOS:-kejun.huang}"; PARTITION="${NOVAE_PARTITION:-}"
TIME="${NOVAE_SPATIAL_TIME:-48:00:00}"; NAME="${NOVAE_SPATIAL_JOB_NAME:-novae_spatial_validation}"
safe_path() { [[ -n "$2" && "$2" != *$'\n'* && "$2" != *$'\r'* && "$2" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "$1 contains unsafe path characters" >&2; exit 2; }; }
safe_val() { [[ -n "$2" && "$2" != *$'\n'* && "$2" != *$'\r'* && "$2" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "$1 contains unsafe scheduler characters" >&2; exit 2; }; }
for pair in "REPO_DIR:${REPO_DIR}" "H5AD:${H5AD}" "POST:${POST}" "RUN_ROOT:${RUN_ROOT}" "OUTPUT:${OUTPUT}" "JOB:${JOB}" "LOG:${LOG}" "ENV_PATH:${ENV_PATH}"; do safe_path "${pair%%:*}" "${pair#*:}"; done
for pair in "ACCOUNT:${ACCOUNT}" "QOS:${QOS}" "TIME:${TIME}" "NAME:${NAME}"; do safe_val "${pair%%:*}" "${pair#*:}"; done
[[ "$TIME" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "TIME must be HH:MM:SS" >&2; exit 2; }
[[ -d "$REPO_DIR" && -f "$REPO_DIR/scripts/run_novae_spatial_biological_validation.py" ]] || { echo "repository/orchestrator not found" >&2; exit 2; }
[[ ! -e "$OUTPUT" && ! -e "$JOB" ]] || { echo "refusing existing output/job" >&2; exit 2; }
[[ "$OUTPUT" != "$H5AD" && "$OUTPUT" != "$POST" ]] || { echo "output must not overwrite an input" >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$LOG"
if [[ -n "$PARTITION" ]]; then safe_val PARTITION "$PARTITION"; PARTITION_LINE="#SBATCH --partition=$PARTITION"; else PARTITION_LINE=""; fi
TMP="${JOB}.partial.$$"; trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${NAME}
#SBATCH --output=${LOG}/validation_%j.out
#SBATCH --error=${LOG}/validation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96gb
#SBATCH --time=${TIME}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
${PARTITION_LINE}
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
[[ -f ${H5AD@Q} && -f ${POST@Q} && ! -e ${OUTPUT@Q} ]]
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate ${ENV_PATH@Q}; set -u
cd ${REPO_DIR@Q}
python scripts/run_novae_spatial_biological_validation.py --novae-h5ad ${H5AD@Q} --post-nmf-obs ${POST@Q} --output-dir ${OUTPUT@Q} --permutations 1000 --seed 42
EOF
chmod 700 "$TMP"; ln "$TMP" "$JOB"; rm -f "$TMP"; trap - EXIT
if (( RENDER_ONLY )); then echo "Rendered $JOB"; exit 0; fi
JOB_ID=$(sbatch "$JOB" | awk '{print $4}'); echo "Queued $JOB_ID; output will be $OUTPUT"
