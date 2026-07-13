#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"
DATA_DIR="/blue/kejun.huang/vasco.hinostroza/data/skin_dataset"
PROCESSED_DIR="${DATA_DIR}/processed"
SOURCE_H5AD="${PROCESSED_DIR}/skin_visium_ssc_1mmfov_spatial.h5ad"
CONDA_ENV="/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4"
TILE=250
DATASET_ID="skin_visium_ssc_${TILE}umfov"
PREP_LOG_DIR="${REPO_DIR}/runs/pseudofov_prep_logs"
PREP_SCRIPT="${PREP_LOG_DIR}/submit_skin_pseudofov_250um_prep.sh"

mkdir -p "${PREP_LOG_DIR}"

cat > "${PREP_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=skin250_prep
#SBATCH --output=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/pseudofov_prep_logs/skin250_prep.out
#SBATCH --error=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/pseudofov_prep_logs/skin250_prep.err
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

REPO_DIR="/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"
DATA_DIR="/blue/kejun.huang/vasco.hinostroza/data/skin_dataset"
PROCESSED_DIR="${DATA_DIR}/processed"
SOURCE_H5AD="${PROCESSED_DIR}/skin_visium_ssc_1mmfov_spatial.h5ad"
CONDA_ENV="/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4"
TILE=250
DATASET_ID="skin_visium_ssc_${TILE}umfov"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

cd "${REPO_DIR}"

python scripts/analyze_skin_pseudo_fov_sizes.py   --h5ad "${SOURCE_H5AD}"   --output-dir "${PROCESSED_DIR}/pseudo_fov_sweep_analysis"   --tile-sizes-um "${TILE}"   --min-spots 20

python scripts/retile_skin_visium_spatial_h5ad.py   --source-h5ad "${SOURCE_H5AD}"   --output-dir "${PROCESSED_DIR}"   --dataset-id "${DATASET_ID}"   --pseudo-fov-tile-um "${TILE}"

CELL2LOC_PRESET="presets/${DATASET_ID}_poisson75_hpg_cell2loc_gpu.json"
NMF_PRESET="presets/${DATASET_ID}_poisson75_hpg_nmf_gpu.json"
POST_NMF_PRESET="presets/${DATASET_ID}_poisson75_hpg_post_nmf_cpu.json"
RCAUSAL_PRESET="presets/${DATASET_ID}_poisson75_hpg_rcausal_cpu.json"
MLP_TUNE_PRESET="presets/${DATASET_ID}_poisson75_hpg_mlp_tune_once_cpu.json"
MLP_EVAL_PRESET="presets/${DATASET_ID}_poisson75_hpg_mlp_eval_fixed_cpu.json"

python run_pipeline.py --config "${CELL2LOC_PRESET}" --validate
python run_pipeline.py --config "${NMF_PRESET}" --validate
python run_pipeline.py --config "${POST_NMF_PRESET}" --validate
python run_pipeline.py --config "${RCAUSAL_PRESET}" --validate
python run_pipeline.py --config "${MLP_TUNE_PRESET}" --validate
python run_pipeline.py --config "${MLP_EVAL_PRESET}" --validate

python run_pipeline.py --config "${CELL2LOC_PRESET}"
python run_pipeline.py --config "${NMF_PRESET}"
python run_pipeline.py --config "${POST_NMF_PRESET}"
python run_pipeline.py --config "${RCAUSAL_PRESET}"
python run_pipeline.py --config "${MLP_TUNE_PRESET}"
python run_pipeline.py --config "${MLP_EVAL_PRESET}"

CELL2LOC_RUN_DIR="runs/${DATASET_ID}_poisson75_fullsweep"
NMF_RUN_DIR="runs/${DATASET_ID}_poisson75_fullsweep_nmf_gpu"
POST_NMF_RUN_DIR="runs/${DATASET_ID}_poisson75_post_nmf_cpu"
RCAUSAL_RUN_DIR="runs/${DATASET_ID}_poisson75_rcausal_cpu"
MLP_TUNE_RUN_DIR="runs/${DATASET_ID}_poisson75_mlp_tune_once_cpu"
MLP_EVAL_RUN_DIR="runs/${DATASET_ID}_poisson75_mlp_eval_fixed_cpu"

C2L=$(sbatch "${CELL2LOC_RUN_DIR}/submit.sh" | awk '{print $4}')
NMF=$(sbatch --dependency=afterok:${C2L} "${NMF_RUN_DIR}/submit.sh" | awk '{print $4}')
PN=$(sbatch --dependency=afterok:${NMF} "${POST_NMF_RUN_DIR}/submit.sh" | awk '{print $4}')
RC=$(sbatch --dependency=afterok:${PN} "${RCAUSAL_RUN_DIR}/submit.sh" | awk '{print $4}')
T1=$(sbatch --dependency=afterok:${PN} "${MLP_TUNE_RUN_DIR}/submit.sh" | awk '{print $4}')
EV=$(sbatch --dependency=afterok:${RC},afterok:${T1} "${MLP_EVAL_RUN_DIR}/submit.sh" | awk '{print $4}')

echo "Queued ${DATASET_ID}: cell2loc=${C2L}, nmf=${NMF}, post_nmf=${PN}, rcausal=${RC}, mlp_tune=${T1}, mlp_eval=${EV}"
echo "Monitor with:"
echo "squeue -u \$USER"
echo "sacct -j ${C2L},${NMF},${PN},${RC},${T1},${EV} --format=JobID,JobName%28,State,ExitCode,Elapsed"
EOF

chmod +x "${PREP_SCRIPT}"
PREP_JOB=$(sbatch "${PREP_SCRIPT}" | awk '{print $4}')
echo "Queued 250um pseudo-FOV prep job: ${PREP_JOB}"
echo "Monitor prep with:"
echo "squeue -j ${PREP_JOB}"
echo "tail -n 200 ${PREP_LOG_DIR}/skin250_prep.out"
