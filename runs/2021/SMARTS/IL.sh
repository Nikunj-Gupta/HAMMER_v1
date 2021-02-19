#!/bin/bash
#SALLOC --mem=5000M
#SALLOC --time=00-1:00
#SALLOC --account=def-mtaylor3
#SALLOC --x11
module load StdEnv/2020 python/3.7.9 sumo/1.7.0 geos/3.8.1
export SUMO_HOME=$EBROOTSUMO
source ~/VENV/bin/activate

tensorboard --logdir=logs/ --host 0.0.0.0 &
scl envision start -s ../../../../SMARTS/scenarios -p 8081 &
time python ../../../hammer-smarts.py --config ../../../configs/2021/cn/hyperparams.yaml --scenario ../../../../SMARTS/benchmark/scenarios/intersections/4lane --expname IL-smarts --hammer 0 --headless True