#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool"
DATA_DIR="/blue/kejun.huang/vasco.hinostroza/data/skin_dataset"
PROCESSED_DIR="${DATA_DIR}/processed"
SOURCE_H5AD="${PROCESSED_DIR}/skin_visium_ssc_1mmfov_spatial.h5ad"
CONDA_ENV="/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4"
TILE_SIZES=(1000 750 500)

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"

python scripts/analyze_skin_pseudo_fov_sizes.py \
  --h5ad "${SOURCE_H5AD}" \
  --output-dir "${PROCESSED_DIR}/pseudo_fov_sweep_analysis" \
  --tile-sizes-um "${TILE_SIZES[@]}" \
  --min-spots 20

declare -A CELL2LOC_JOBS
declare -A NMF_JOBS
declare -A DOWNSTREAM_JOBS

for TILE in "${TILE_SIZES[@]}"; do
  DATASET_ID="skin_visium_ssc_${TILE}umfov"

  echo "=== Building ${DATASET_ID} ==="
  python scripts/retile_skin_visium_spatial_h5ad.py \
    --source-h5ad "${SOURCE_H5AD}" \
    --output-dir "${PROCESSED_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --pseudo-fov-tile-um "${TILE}"

  CELL2LOC_PRESET="presets/${DATASET_ID}_poisson75_hpg_cell2loc_gpu.json"
  NMF_PRESET="presets/${DATASET_ID}_poisson75_hpg_nmf_gpu.json"
  DOWNSTREAM_PRESET="presets/${DATASET_ID}_poisson75_hpg_downstream_cpu.json"

  python run_pipeline.py --config "${CELL2LOC_PRESET}" --validate
  python run_pipeline.py --config "${NMF_PRESET}" --validate
  python run_pipeline.py --config "${DOWNSTREAM_PRESET}" --validate

  python run_pipeline.py --config "${CELL2LOC_PRESET}"
  python run_pipeline.py --config "${NMF_PRESET}"
  python run_pipeline.py --config "${DOWNSTREAM_PRESET}"

  CELL2LOC_RUN_DIR="runs/${DATASET_ID}_poisson75_fullsweep"
  NMF_RUN_DIR="runs/${DATASET_ID}_poisson75_fullsweep_nmf_gpu"
  DOWNSTREAM_RUN_DIR="runs/${DATASET_ID}_poisson75_fullsweep_downstream_cpu"

  CELL2LOC_JOBS["${TILE}"]=$(sbatch "${CELL2LOC_RUN_DIR}/submit.sh" | awk '{print $4}')
  NMF_JOBS["${TILE}"]=$(sbatch --dependency=afterok:${CELL2LOC_JOBS[${TILE}]} "${NMF_RUN_DIR}/submit.sh" | awk '{print $4}')
  DOWNSTREAM_JOBS["${TILE}"]=$(sbatch --dependency=afterok:${NMF_JOBS[${TILE}]} "${DOWNSTREAM_RUN_DIR}/submit.sh" | awk '{print $4}')

  echo "Queued ${DATASET_ID}: cell2loc=${CELL2LOC_JOBS[${TILE}]}, nmf=${NMF_JOBS[${TILE}]}, downstream=${DOWNSTREAM_JOBS[${TILE}]}"
done

echo
echo "=== Sweep queued ==="
for TILE in "${TILE_SIZES[@]}"; do
  echo "${TILE}um: cell2loc=${CELL2LOC_JOBS[${TILE}]}, nmf=${NMF_JOBS[${TILE}]}, downstream=${DOWNSTREAM_JOBS[${TILE}]}"
done

echo
echo "Monitor with:"
echo "squeue -u \$USER"
echo "sacct -j $(printf "%s,%s,%s," "${CELL2LOC_JOBS[@]}" "${NMF_JOBS[@]}" "${DOWNSTREAM_JOBS[@]}" | sed 's/,$//') --format=JobID,JobName%28,State,ExitCode,Elapsed"
