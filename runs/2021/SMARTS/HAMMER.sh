#!/bin/bash
#SBATCH --mem=5000M
#SBATCH --time=00-10:00
#SBATCH --account=def-mtaylor3
module load StdEnv/2020 python/3.7.9 sumo/1.7.0 geos/3.8.1
export SUMO_HOME=$EBROOTSUMO
source ~/VENV/bin/activate

tensorboard --logdir=logs/ --host 0.0.0.0 &
time python ../../../hammer-smarts.py --config ../../../configs/2021/cn/hyperparams.yaml --scenario ../../../../SMARTS/benchmark/scenarios/intersections/4lane --expname hammer-smarts --hammer 1 --headless True