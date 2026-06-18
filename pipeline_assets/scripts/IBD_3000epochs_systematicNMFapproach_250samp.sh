#!/bin/bash
# Deprecated standalone launcher retained for reference. New runs should use
# presets plus run_pipeline.py instead of calling this script directly.
#SBATCH --job-name=cell2location_IBD_3000_systematicNMF_500samples
#SBATCH --output=/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs250samp/cell2location_IBD_3000ep_systematicNMF_500samp.out
#SBATCH --error=/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/Outputs250samp/cell2location_IBD_3000ep_systematicNMF_500samp.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32  # Adjust based on the number of CPUs you want to use
#SBATCH --nodes=1           # Number of nodes
#SBATCH --time=72:00:00     # Max time (hh:mm:ss)
#SBATCH --mem=500gb           # Memory per node
#SBATCH --qos=kejun.huang-b   # Partition to submit to
#SBATCH --mail-user=tan.m@ufl.edu
#SBATCH --mail-type=ALL


# Load the modules
module load conda

# Activate the Conda environment
conda activate /blue/kejun.huang/tan.m/cell2location_cuda118_torch22

echo "Running cell2location and NMF analysis on IBD data"

# Run the R script
python /blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/IBD_3000epochs_systematicNMFapproach_250samp.py
