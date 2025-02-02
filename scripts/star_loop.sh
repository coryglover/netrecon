#!/bin/bash

for N in $(seq 2 100); do

sbatch -J star_$N star_execute.sh $N;

done