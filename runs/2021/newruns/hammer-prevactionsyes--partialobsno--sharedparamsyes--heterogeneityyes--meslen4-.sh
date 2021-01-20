#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-4:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 50000 --expname hammer-prevactionsyes--partialobsno--sharedparamsyes--heterogeneityyes--meslen4- --hammer 1 --meslen 4 --prevactions 1 --partialobs 0 --sharedparams 1 --heterogeneity 1