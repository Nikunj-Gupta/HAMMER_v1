#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-10:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../hammer-cn.py --config ../../../configs/2021/cn/hyperparams.yaml --maxepisodes 50000 --expname IL-partialobsyes--sharedparamsno- --hammer 0 --randomseed 685 --meslen 4 --prevactions 0 --partialobs 1 --sharedparams 0