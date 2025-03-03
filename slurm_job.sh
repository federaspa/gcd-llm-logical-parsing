#!/bin/bash
#SBATCH --job-name=GCLLM_ministral
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_mig
#SBATCH --time=20:00:00
#SBATCH --output=./%x_%A_%a.out

module load 2024

. /etc/bashrc
. ~/.bashrc

source ~/GCLLM/.venv/bin/activate

echo $$

srun python ~/GCLLM/scripts/logic_problems.py --models-path /home/fraspanti/LLMs/ --dataset-name GSM8K_symbolic --shots-number 5shots --n-gpu-layers -1 --model-name ministral