#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=00-30:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../multiwalker.py --config ../../../configs/2021/mw/hyperparameters.yaml --hammer 1 --expname mw-hammer-rs--nagents-59-meslen-4 --nwalkers 5 --maxepisodes 100000 --randomseed 9 --meslen 4