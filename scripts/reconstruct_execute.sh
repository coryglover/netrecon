#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=infinite
#SBATCH --job-name=Reconstruct_Network
#SBATCH --output=/scratch/glover.co/netrecon/reconstruct/out/output_%A_%a.out
#SBATCH --error=/scratch/glover.co/netrecon/reconstruct/err/error_%A_%a.err
#SBATCH --array=1-800%50

# Read the correct line from params.txt
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" recon_params.txt)

echo "Running job with parameters: ${PARAMS}"

# Run experiment
python reconstruct.py $PARAMS
