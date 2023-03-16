#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/scripts/evaluate/SiN_v1.0/gemnet-T.yml
#EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp-evaluate/SiN_v1.0/GemNet-T
#EXPID=simulation_test

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode evaluate \
    --config-yml $CONFIG \
    --checkpoint-path $2

#    --run-dir $EXPDIR \
#    --identifier $EXPID \

cd $CURRENT_PATH

