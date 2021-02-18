#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-10:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../il_multiwalker.py --config ../../../configs/2021/cn/hyperparams.yaml --expname il_multiwalker--mw--rs10 --randomseed 10