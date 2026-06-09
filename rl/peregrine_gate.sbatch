#!/bin/bash
#SBATCH --job-name=peregrine_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/network/fl3689/peregrine_gate.out
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash /scratch/network/fl3689/run_gate_body.sh
echo "SBATCH_GATE_DONE rc=$?"
