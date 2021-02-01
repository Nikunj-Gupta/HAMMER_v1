#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammerrandommes-prevactionsno--partialobsyes--sharedparamsyes--heterogeneityno--discretemes--rs879--meslen2- --hammer 1 --randomseed 879 --discretemes 1 --meslen 2 --prevactions 0 --partialobs 1 --sharedparams 1 --heterogeneity 0