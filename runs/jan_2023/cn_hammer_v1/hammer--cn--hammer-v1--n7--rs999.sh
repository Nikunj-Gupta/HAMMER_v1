#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=24:00:00
#SBATCH --mem=40GB

source ../venvs/hammer/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

time python hammer-cn.py --config configs/2021/cn/hyperparams.yaml --nagents  7 --expname hammer--cn--hammer-v1--n7--rs999 --discretemes 0 --meslen 1  --randomseed 999 --randommes 0  --hammer 1 