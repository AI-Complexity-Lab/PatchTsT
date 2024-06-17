#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=47GB
#SBATCH --account=alrodri0

# set up job
module load python/3.9.12 cuda
pushd /home/liruipu/CDC-FluSight-2023/
source env/bin/activate

# run online training
cd src/forecaster
python online_training.py --input=12
