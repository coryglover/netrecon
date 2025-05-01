#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=12:00:00
#SBATCH --job-name=PA_Network
#SBATCH --output=/scratch/glover.co/netrecon/pa/out/output_%A_%a.out
#SBATCH --error=/scratch/glover.co/netrecon/pa/err/error_%A_%a.err
#SBATCH --array=1-1000%10

# Read the correct line from params.txt
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pa_params_2.txt)

echo "Running job with parameters: ${PARAMS}"

# Run experiment
python create_network.py $PARAMS

# done
