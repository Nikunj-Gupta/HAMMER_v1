#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-20:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 100000 --expname hammer-prevactionsyes--partialobsyes--sharedparamsno--heterogeneityno--discretemes--rs10--meslen1- --hammer 1 --randomseed 10 --maxcycles 100 --limit 10 --discretemes 1 --meslen 1 --prevactions 1 --partialobs 1 --sharedparams 0 --heterogeneity 0