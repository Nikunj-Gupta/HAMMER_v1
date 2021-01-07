#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=2:00:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../main_complete_state_and_prev_actions.py --config ../../../configs/2021/cn/hyperparams.yaml --hammer 1 --expname cn-hammer-rs-10-meslen-4 --maxepisodes 100000 --randomseed 10 --meslen 4