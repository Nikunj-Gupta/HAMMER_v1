#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-8:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammerrandommes-prevactionsyes--partialobsyes--sharedparamsyes--heterogeneityno--continuousmes--rs465--meslen8- --hammer 1 --randomseed 465 --discretemes 0 --meslen 8 --prevactions 1 --partialobs 1 --sharedparams 1 --heterogeneity 0