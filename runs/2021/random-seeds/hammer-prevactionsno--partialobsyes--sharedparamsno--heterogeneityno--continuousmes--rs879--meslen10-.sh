#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammer-prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--continuousmes--rs879--meslen10- --hammer 1 --randomseed 879 --discretemes 0 --meslen 10 --prevactions 0 --partialobs 1 --sharedparams 0 --heterogeneity 0