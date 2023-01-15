#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=10:00:00
#SBATCH --mem=40GB

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

time python hammer-cn.py --config configs/2021/cn/hyperparams.yaml --nagents  5 --expname hammer--cn--randommes--n5--rs465 --discretemes 0 --meslen 1  --randomseed 465 --randommes 1  --hammer 1 