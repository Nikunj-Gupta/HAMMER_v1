#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python hammer-cn.py --config configs/2021/cn/hyperparams.yaml --nagents  7 --expname hammer--cn--randommes--n7--rs674 --discretemes 0 --meslen 1  --randomseed 674 --randommes 1  --hammer 1 