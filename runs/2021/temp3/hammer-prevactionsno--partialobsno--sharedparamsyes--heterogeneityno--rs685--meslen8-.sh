#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-4:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 50000 --expname hammer-prevactionsno--partialobsno--sharedparamsyes--heterogeneityno--rs685--meslen8- --hammer 1 --randomseed 685 --meslen 8 --prevactions 0 --partialobs 0 --sharedparams 1 --heterogeneity 0