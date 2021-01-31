#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammer-prevactionsno--partialobsno--sharedparamsno--heterogeneityno--continuousmes--rs984--meslen4- --hammer 1 --randomseed 984 --discretemes 0 --meslen 4 --prevactions 0 --partialobs 0 --sharedparams 0 --heterogeneity 0