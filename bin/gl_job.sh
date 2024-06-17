#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-09:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=47GB
#SBATCH --account=alrodri0
#SBATCH --mail-user=zhicao@umich.edu
#SBATCH --mail-type=END

# set up job
pushd /home/zhicao/CDC-FluSight-2023/
source /home/zhicao/miniconda3/bin/activate
conda activate py3

# run online training
cd src/forecaster
python online_training.py --input $1
