#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-7:30
#SBATCH --account=def-mtaylor3
source ~/VENV/bin/activate

tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-sr.py --config ../../../configs/2021/cn/hyperparams.yaml --expname HAMMER-sr --hammer 1