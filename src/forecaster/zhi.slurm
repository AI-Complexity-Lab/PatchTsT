#!/bin/bash
#SBATCH --job-name=x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-user=zhicao@umich.edu
#SBATCH --mail-type=END
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=40G
#SBATCH --output=my_training_job_%j.out

nvidia-smi

python online_training.py -i=1