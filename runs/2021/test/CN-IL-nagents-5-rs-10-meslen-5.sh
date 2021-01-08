#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=00-30:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../main_complete_state_and_prev_actions.py --config ../../../configs/2021/cn/hyperparams.yaml --hammer 0 --expname CN-IL-nagents-5-rs-10-meslen-5 --nagents 5 --maxepisodes 100000 --randomseed 10 --meslen 5