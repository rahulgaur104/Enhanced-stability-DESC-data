#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=72G
#SBATCH --time=01:00:00

module purge;\

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93

module load anaconda3/2023.3;\

conda activate desc-env;\

#python3 calculate_OP_ball3_gamma.py
python3 calculate_OT_ball3_gamma.py
#python3 calculate_OH_ball5_gamma.py


