#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=48:00:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python3 ../../../main_complete_state_and_prev_actions.py  --config ../../../configs/2021/cn/hyperparams.yaml --expname test --hammer 1 --maxepisodes 50000 --meslen 2 