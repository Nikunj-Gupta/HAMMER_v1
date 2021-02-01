#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammerrandommes-prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--discretemes--rs685--meslen2- --hammer 1 --randomseed 685 --randommes 1 --limit 10 --discretemes 1 --meslen 2 --prevactions 0 --partialobs 1 --sharedparams 0 --heterogeneity 0