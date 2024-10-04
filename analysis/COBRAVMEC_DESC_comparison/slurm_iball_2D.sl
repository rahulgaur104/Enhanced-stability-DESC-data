#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00

module purge;
module use /home/caoxiang/module;
module load stellopt/intel;

export PATH="/home/caoxiang/share/stellopt_develop_intel/bin:$PATH"


#srun -n64 xcobravmec cobra.input_og
srun -n48 xcobravmec cobra.input_HELIOTRON

