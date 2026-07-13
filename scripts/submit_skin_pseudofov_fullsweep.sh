#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"
DATA_DIR="/blue/kejun.huang/vasco.hinostroza/data/skin_dataset"
PROCESSED_DIR="${DATA_DIR}/processed"
SOURCE_H5AD="${PROCESSED_DIR}/skin_visium_ssc_1mmfov_spatial.h5ad"
CONDA_ENV="/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4"
TILE_SIZES=(1000 750 500)
PREP_LOG_DIR="${REPO_DIR}/runs/pseudofov_prep_logs"
PREP_SCRIPT="${PREP_LOG_DIR}/submit_skin_pseudofov_fullsweep_prep.sh"

mkdir -p "${PREP_LOG_DIR}"

cat > "${PREP_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=skin_pfs_prep
#SBATCH --output=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/pseudofov_prep_logs/skin_pfs_prep.out
#SBATCH --error=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/pseudofov_prep_logs/skin_pfs_prep.err
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
TILE_SIZES=(1000 750 500)

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

cd "${REPO_DIR}"

python scripts/analyze_skin_pseudo_fov_sizes.py   --h5ad "${SOURCE_H5AD}"   --output-dir "${PROCESSED_DIR}/pseudo_fov_sweep_analysis"   --tile-sizes-um "${TILE_SIZES[@]}"   --min-spots 20

declare -A CELL2LOC_JOBS

declare -A NMF_JOBS

declare -A POST_NMF_JOBS

declare -A RCAUSAL_JOBS

declare -A MLP_TUNE_JOBS

declare -A MLP_EVAL_JOBS

for TILE in "${TILE_SIZES[@]}"; do
  DATASET_ID="skin_visium_ssc_${TILE}umfov"

  echo "=== Building ${DATASET_ID} ==="
  python scripts/retile_skin_visium_spatial_h5ad.py     --source-h5ad "${SOURCE_H5AD}"     --output-dir "${PROCESSED_DIR}"     --dataset-id "${DATASET_ID}"     --pseudo-fov-tile-um "${TILE}"

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

  CELL2LOC_JOBS["${TILE}"]=$(sbatch "${CELL2LOC_RUN_DIR}/submit.sh" | awk '{print $4}')
  NMF_JOBS["${TILE}"]=$(sbatch --dependency=afterok:${CELL2LOC_JOBS[${TILE}]} "${NMF_RUN_DIR}/submit.sh" | awk '{print $4}')
  POST_NMF_JOBS["${TILE}"]=$(sbatch --dependency=afterok:${NMF_JOBS[${TILE}]} "${POST_NMF_RUN_DIR}/submit.sh" | awk '{print $4}')
  RCAUSAL_JOBS["${TILE}"]=$(sbatch --dependency=afterok:${POST_NMF_JOBS[${TILE}]} "${RCAUSAL_RUN_DIR}/submit.sh" | awk '{print $4}')
  MLP_TUNE_JOBS["${TILE}"]=$(sbatch --dependency=afterok:${POST_NMF_JOBS[${TILE}]} "${MLP_TUNE_RUN_DIR}/submit.sh" | awk '{print $4}')
  MLP_EVAL_JOBS["${TILE}"]=$(sbatch --dependency=afterok:${MLP_TUNE_JOBS[${TILE}]},afterok:${RCAUSAL_JOBS[${TILE}]} "${MLP_EVAL_RUN_DIR}/submit.sh" | awk '{print $4}')

  echo "Queued ${DATASET_ID}: cell2loc=${CELL2LOC_JOBS[${TILE}]}, nmf=${NMF_JOBS[${TILE}]}, post_nmf=${POST_NMF_JOBS[${TILE}]}, rcausal=${RCAUSAL_JOBS[${TILE}]}, mlp_tune=${MLP_TUNE_JOBS[${TILE}]}, mlp_eval=${MLP_EVAL_JOBS[${TILE}]}"
done

echo
echo "=== Sweep queued ==="
for TILE in "${TILE_SIZES[@]}"; do
  echo "${TILE}um: cell2loc=${CELL2LOC_JOBS[${TILE}]}, nmf=${NMF_JOBS[${TILE}]}, post_nmf=${POST_NMF_JOBS[${TILE}]}, rcausal=${RCAUSAL_JOBS[${TILE}]}, mlp_tune=${MLP_TUNE_JOBS[${TILE}]}, mlp_eval=${MLP_EVAL_JOBS[${TILE}]}"
done

echo
echo "Monitor with:"
echo "squeue -u \$USER"
JOB_LIST=$(printf "%s,%s,%s,%s,%s,%s," "${CELL2LOC_JOBS[@]}" "${NMF_JOBS[@]}" "${POST_NMF_JOBS[@]}" "${RCAUSAL_JOBS[@]}" "${MLP_TUNE_JOBS[@]}" "${MLP_EVAL_JOBS[@]}" | sed 's/,$//')
echo "sacct -j ${JOB_LIST} --format=JobID,JobName%28,State,ExitCode,Elapsed"
EOF

chmod +x "${PREP_SCRIPT}"
PREP_JOB=$(sbatch "${PREP_SCRIPT}" | awk '{print $4}')
echo "Queued skin pseudo-FOV prep job: ${PREP_JOB}"
echo "Monitor prep with:"
echo "squeue -j ${PREP_JOB}"
echo "tail -n 200 ${PREP_LOG_DIR}/skin_pfs_prep.out"
