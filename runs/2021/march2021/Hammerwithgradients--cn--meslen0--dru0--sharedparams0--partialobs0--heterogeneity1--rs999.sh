#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-with-gradient.py --config ../../../configs/2021/cn/hyperparams.yaml --expname Hammerwithgradients--cn--meslen0--dru0--sharedparams0--partialobs0--heterogeneity1--rs999 --partialobs 0 --heterogeneity 1 --dru_toggle 0 --meslen 0 --randomseed 999 --sharedparams 0