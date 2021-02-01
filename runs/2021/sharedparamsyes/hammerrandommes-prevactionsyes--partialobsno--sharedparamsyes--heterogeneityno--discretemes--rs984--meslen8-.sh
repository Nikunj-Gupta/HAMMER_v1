#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammerrandommes-prevactionsyes--partialobsno--sharedparamsyes--heterogeneityno--discretemes--rs984--meslen8- --hammer 1 --randomseed 984 --discretemes 1 --meslen 8 --prevactions 1 --partialobs 0 --sharedparams 1 --heterogeneity 0