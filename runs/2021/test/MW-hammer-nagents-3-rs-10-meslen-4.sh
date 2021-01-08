#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=00-30:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../multiwalker.py --config ../../../configs/2021/mw/hyperparameters.yaml --hammer 1 --expname MW-hammer-nagents-3-rs-10-meslen-4 --nagents 3 --maxepisodes 100000 --randomseed 10 --meslen 4