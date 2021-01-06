#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=48:00:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../main_complete_state_and_prev_actions.py  --config ../../configs/2021/cn/hyperparams.yaml --randomseed 24 --expname hammer-trial-1 --meslen 1 --hammer 1 