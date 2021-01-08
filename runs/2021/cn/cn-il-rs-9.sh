#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=00-06:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../main_complete_state_and_prev_actions.py --config ../../../configs/2021/cn/hyperparams.yaml --hammer 0 --expname cn-il-rs-9 --maxepisodes 100000 --randomseed 9