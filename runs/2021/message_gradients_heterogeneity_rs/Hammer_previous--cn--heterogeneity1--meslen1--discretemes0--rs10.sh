#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --expname Hammer_previous--cn--heterogeneity1--meslen1--discretemes0--rs10 --partialobs 0 --heterogeneity 1 --randomseed 10 --nagents 3 --meslen 1 --discretemes 0 --limit 10