#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-4:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 50000 --expname hammer-prevactionsyes--partialobsyes--sharedparamsyes--heterogeneityno--continuousmes--meslen6- --hammer 1 --discretemes 0 --meslen 6 --prevactions 1 --partialobs 1 --sharedparams 1 --heterogeneity 0