#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-with-gradient.py --config ../../../configs/2021/cn/hyperparams.yaml --expname Hammerwithgradients--cn--meslen1--dru1--sharedparams0--partialobs1--heterogeneity0--rs999 --partialobs 1 --heterogeneity 0 --dru_toggle 1 --meslen 1 --randomseed 999 --sharedparams 0