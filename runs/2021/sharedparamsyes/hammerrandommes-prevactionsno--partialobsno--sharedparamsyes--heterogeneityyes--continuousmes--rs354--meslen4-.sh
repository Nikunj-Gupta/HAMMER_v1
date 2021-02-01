#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammerrandommes-prevactionsno--partialobsno--sharedparamsyes--heterogeneityyes--continuousmes--rs354--meslen4- --hammer 1 --randomseed 354 --discretemes 0 --meslen 4 --prevactions 0 --partialobs 0 --sharedparams 1 --heterogeneity 1