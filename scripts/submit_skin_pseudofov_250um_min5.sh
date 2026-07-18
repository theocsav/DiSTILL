#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"
PREP_LOG_DIR="${REPO_DIR}/runs/pseudofov_prep_logs"
PREP_SCRIPT="${PREP_LOG_DIR}/submit_skin_pseudofov_250um_min5_prep.sh"
mkdir -p "${PREP_LOG_DIR}"

cat > "${PREP_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
# This exploratory arm reuses the completed 250um NMF output and tests
# downstream FOV eligibility at five total cells per FOV.
# Neighborhood enrichment still requires at least two valid spatial cells.
#SBATCH --job-name=skin250_m5
#SBATCH --output=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/pseudofov_prep_logs/skin250_m5.out
#SBATCH --error=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/pseudofov_prep_logs/skin250_m5.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --account=kejun.huang
#SBATCH --qos=kejun.huang-b
#SBATCH --mail-user=vasco.hinostroza@ufl.edu
#SBATCH --mail-type=ALL
set -euo pipefail
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4"
set -u
cd "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"

python run_pipeline.py --config presets/skin_visium_ssc_250umfov_poisson75_hpg_post_nmf_cpu_min5.json --validate
python run_pipeline.py --config presets/skin_visium_ssc_250umfov_poisson75_hpg_mlp_tune_once_cpu_min5.json --validate
python run_pipeline.py --config presets/skin_visium_ssc_250umfov_poisson75_hpg_mlp_eval_fixed_cpu_min5.json --validate

python run_pipeline.py --config presets/skin_visium_ssc_250umfov_poisson75_hpg_post_nmf_cpu_min5.json
python run_pipeline.py --config presets/skin_visium_ssc_250umfov_poisson75_hpg_mlp_tune_once_cpu_min5.json
python run_pipeline.py --config presets/skin_visium_ssc_250umfov_poisson75_hpg_mlp_eval_fixed_cpu_min5.json

PN=$(sbatch runs/skin_visium_ssc_250umfov_poisson75_post_nmf_cpu_min5/submit.sh | awk '{print $4}')
T1=$(sbatch --dependency=afterok:${PN} runs/skin_visium_ssc_250umfov_poisson75_mlp_tune_once_cpu_min5/submit.sh | awk '{print $4}')
EV=$(sbatch --dependency=afterok:${PN},afterok:${T1} runs/skin_visium_ssc_250umfov_poisson75_mlp_eval_fixed_cpu_min5/submit.sh | awk '{print $4}')
echo "Queued 250um min5 exploratory classification: post_nmf=${PN}, mlp_tune=${T1}, mlp_eval=${EV}"
echo "sacct -j ${PN},${T1},${EV} --format=JobID,JobName%28,State,ExitCode,Elapsed"
EOF

chmod +x "${PREP_SCRIPT}"
PREP_JOB=$(sbatch "${PREP_SCRIPT}" | awk '{print $4}')
echo "Queued 250um min5 exploratory prep job: ${PREP_JOB}"
echo "tail -n 200 ${PREP_LOG_DIR}/skin250_m5.out"
