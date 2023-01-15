#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python hammer-cn.py --config configs/2021/cn/hyperparams.yaml --nagents  3 --expname hammer--cn--randommes--n3--rs354 --discretemes 0 --meslen 1  --randomseed 354 --randommes 1  --hammer 1 