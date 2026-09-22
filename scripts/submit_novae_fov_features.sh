#!/usr/bin/env bash
# Render/submit the CPU-only NOVAE historical-164 feature adapter.
set -euo pipefail
RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" || "${1:-}" == "--no-submit" ]]; then RENDER_ONLY=1; shift; fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--render-only|--no-submit]" >&2; exit 2; }
REPO_DIR="${NOVAE_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
BASE_H5AD="${NOVAE_FOV_BASE_H5AD:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/cosmx_with_nmf.h5ad}"
NOVAE_H5AD="${NOVAE_FOV_NOVAE_H5AD:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot/paired_cpu_diagnostic/skin_visium_ssc_paired_cpu_calibrated/novae_skin_visium_ssc_paired_cpu_calibrated_zero_shot.h5ad}"
FEATURE_DIR="${NOVAE_FOV_FEATURE_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_inputs}"
SOURCE_DIR="${NOVAE_FOV_SOURCE_OUTPUT_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs}"
RUN_ROOT="${NOVAE_FOV_RUN_ROOT:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_res1_fov_features_historical164}"
OUTPUT_DIR="${NOVAE_FOV_OUTPUT_DIR:-${RUN_ROOT}/features}"
LOG_DIR="${NOVAE_FOV_LOG_DIR:-${RUN_ROOT}/logs}"
JOB_SCRIPT="${NOVAE_FOV_JOB_SCRIPT:-${RUN_ROOT}/submit_novae_fov_features.sbatch}"
CONDA_ENV="${NOVAE_FOV_CONDA_ENV:-/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4}"
ACCOUNT="${NOVAE_ACCOUNT:-kejun.huang}"; QOS="${NOVAE_QOS:-kejun.huang}"; PARTITION="${NOVAE_PARTITION:-}"
TIME_LIMIT="${NOVAE_FOV_TIME:-08:00:00}"
reject_conflict() { local n="$1" expected="$2" actual="${!1-}"; [[ -z "$actual" || "$actual" == "$expected" ]] || { echo "$n conflicts with the frozen FOV adapter contract" >&2; exit 2; }; }
reject_conflict NOVAE_PRIMARY_RESOLUTION "1.0"
reject_conflict NOVAE_DOMAIN_KEY "novae_domains_res1.0"
reject_conflict NOVAE_VALIDITY_KEY "neighborhood_valid"
reject_conflict NOVAE_ACCELERATOR "cpu"
reject_conflict NOVAE_CPUS_PER_TASK "2"
validate() { local n="$1" v="$2"; [[ -n "$v" && "$v" != *$'\n'* && "$v" != *$'\r'* && "$v" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "$n contains unsafe characters" >&2; exit 2; }; }
for p in "NOVAE_REPO_DIR:$REPO_DIR" "NOVAE_FOV_BASE_H5AD:$BASE_H5AD" "NOVAE_FOV_NOVAE_H5AD:$NOVAE_H5AD" "NOVAE_FOV_FEATURE_DIR:$FEATURE_DIR" "NOVAE_FOV_SOURCE_OUTPUT_DIR:$SOURCE_DIR" "NOVAE_FOV_RUN_ROOT:$RUN_ROOT" "NOVAE_FOV_OUTPUT_DIR:$OUTPUT_DIR" "NOVAE_FOV_LOG_DIR:$LOG_DIR" "NOVAE_FOV_JOB_SCRIPT:$JOB_SCRIPT" "NOVAE_FOV_CONDA_ENV:$CONDA_ENV"; do validate "${p%%:*}" "${p#*:}"; done
for p in "NOVAE_ACCOUNT:$ACCOUNT" "NOVAE_QOS:$QOS" "NOVAE_FOV_TIME:$TIME_LIMIT"; do validate "${p%%:*}" "${p#*:}"; done
[[ "$TIME_LIMIT" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "NOVAE_FOV_TIME must use HH:MM:SS" >&2; exit 2; }
[[ "$BASE_H5AD" != "$NOVAE_H5AD" && "$OUTPUT_DIR" != "$BASE_H5AD" && "$OUTPUT_DIR" != "$NOVAE_H5AD" ]] || { echo "input/output paths conflict" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" && ! -e "$JOB_SCRIPT" ]] || { echo "refusing existing output or job script" >&2; exit 2; }
if [[ -n "$PARTITION" ]]; then validate NOVAE_PARTITION "$PARTITION"; PARTITION_DIRECTIVE="#SBATCH --partition=$PARTITION"; else PARTITION_DIRECTIVE=""; fi
mkdir -p "$RUN_ROOT" "$LOG_DIR"
TMP="$JOB_SCRIPT.partial.$$"; trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=novae_fov_features
#SBATCH --output=$LOG_DIR/novae_fov_features_%j.out
#SBATCH --error=$LOG_DIR/novae_fov_features_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=96gb
#SBATCH --time=$TIME_LIMIT
#SBATCH --account=$ACCOUNT
#SBATCH --qos=$QOS
$PARTITION_DIRECTIVE
set -euo pipefail
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$CONDA_ENV"; set -u
cd "$REPO_DIR"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing existing output: $OUTPUT_DIR" >&2; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="void"
python scripts/build_novae_fov_features.py --base-h5ad "$BASE_H5AD" --novae-h5ad "$NOVAE_H5AD" --feature-dir "$FEATURE_DIR" --source-output-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" --expected-domains L0,L1,L2,L3,L4,L5,L6,L7,L8
EOF
chmod +x "$TMP"; mv "$TMP" "$JOB_SCRIPT"; trap - EXIT
if (( RENDER_ONLY )); then echo "Rendered sbatch script: $JOB_SCRIPT"; exit 0; fi
JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}'); echo "Queued NOVAE FOV feature adapter: $JOB_ID"; echo "Outputs: $OUTPUT_DIR"
