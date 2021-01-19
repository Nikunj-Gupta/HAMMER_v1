#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-4:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 50000 --expname hammer-prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--rs465--meslen4- --hammer 1 --randomseed 465 --meslen 4 --prevactions 0 --partialobs 1 --sharedparams 0 --heterogeneity 0