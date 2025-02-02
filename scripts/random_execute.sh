#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=infinite
#SBATCH --job-name=Random_Network
#SBATCH --output=out/random.out
#SBATCH --error=err/random.err

source activate /home/glover.co/miniconda3/envs/gt

for avg_k in $(seq 1 20) do

# Create network
mkdir -p /work/ccnr/glover.co/net_reconstruction/netrecon/data/random/