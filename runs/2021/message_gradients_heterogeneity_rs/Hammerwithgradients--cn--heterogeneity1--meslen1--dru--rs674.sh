#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-with-gradient.py --config ../../../configs/2021/cn/hyperparams.yaml --expname Hammerwithgradients--cn--heterogeneity1--meslen1--dru--rs674 --partialobs 0 --heterogeneity 1 --randomseed 674 --nagents 3 --meslen 1 --envname cn --dru_toggle 1 --limit 10