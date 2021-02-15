#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-with-gradient.py --config ../../../configs/2021/cn/hyperparams.yaml --expname Hammerwithgradients--cn--partialobs0--meslen1--dru--rs234 --partialobs 0 --randomseed 234 --nagents 3 --meslen 1 --envname cn