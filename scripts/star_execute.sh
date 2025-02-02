#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=infinite
#SBATCH --job-name=Star_Network
#SBATCH --output=out/star.out
#SBATCH --error=err/star.err

source activate /home/glover.co/miniconda3/envs/gt
# for i in $(seq 2 100); do

# Make directory
mkdir -p /work/ccnr/glover.co/net_reconstruction/netrecon/data/star/$1

# Run experiment
python star_network.py --file /work/ccnr/glover.co/net_reconstruction/netrecon/data/star/$1 --N $1

# done
