#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-16:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
time python centralized_training.py --config configs/2021/cn/hyperparams.yaml --nagents  10 --expname hammer--cn--coil--n10--rs879 --randomseed 879