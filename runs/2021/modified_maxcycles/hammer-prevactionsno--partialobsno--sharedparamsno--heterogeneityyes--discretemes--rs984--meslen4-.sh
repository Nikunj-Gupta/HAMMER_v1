#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-20:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammer-prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--discretemes--rs984--meslen4- --hammer 1 --randomseed 984 --maxcycles 100 --limit 10 --discretemes 1 --meslen 4 --prevactions 0 --partialobs 0 --sharedparams 0 --heterogeneity 1