#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-10:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-with-gradient-mw.py --config ../../../configs/2021/cn/hyperparams.yaml --expname Hammerwithgradients_mw--mw--meslen1--nodru--rs674 --randomseed 674 --meslen 1 --dru_toggle 0