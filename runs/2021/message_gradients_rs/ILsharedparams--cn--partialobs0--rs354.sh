#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../il.py --config ../../../configs/2021/cn/hyperparams.yaml --expname ILsharedparams--cn--partialobs0--rs354 --partialobs 0 --randomseed 354 --nagents 3 --envname cn