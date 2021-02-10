#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../il.py --config ../../../configs/2021/cn/hyperparams.yaml --expname ILsharedparams--sr--partialobs1--rs347--new --envname sr --partialobs 1 --randomseed 347 --limit 11 --nagents 2