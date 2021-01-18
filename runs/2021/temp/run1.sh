#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=00-30:00
#SBATCH --account=def-mtaylor3
tensorboard --logdir=logs/ --host 0.0.0.0 &
python ../../../hammer-cn-sharedparams.py --config ../../../configs/2021/cn/hyperparams.yaml --hammer 0 --expname il-cn-sharedparams --nagents 3 --maxepisodes 30000 --randomseed 10 
