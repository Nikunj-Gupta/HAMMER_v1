#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-with-gradient.py --config ../../../configs/2021/cn/hyperparams.yaml --expname Hammerwithgradients--sr--partialobs1--rs984--nodru--meslen4 --envname sr --partialobs 1 --randomseed 984 --limit 11 --nagents 2 --meslen 4