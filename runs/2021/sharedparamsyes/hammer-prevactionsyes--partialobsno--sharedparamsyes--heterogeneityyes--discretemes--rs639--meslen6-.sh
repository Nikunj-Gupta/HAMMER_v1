#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammer-prevactionsyes--partialobsno--sharedparamsyes--heterogeneityyes--discretemes--rs639--meslen6- --hammer 1 --randomseed 639 --discretemes 1 --meslen 6 --prevactions 1 --partialobs 0 --sharedparams 1 --heterogeneity 1