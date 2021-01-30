#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-20:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammer-prevactionsyes--partialobsno--sharedparamsno--heterogeneityyes--continuousmes--rs984--meslen8- --hammer 1 --randomseed 984 --maxcycles 100 --limit 10 --discretemes 0 --meslen 8 --prevactions 1 --partialobs 0 --sharedparams 0 --heterogeneity 1