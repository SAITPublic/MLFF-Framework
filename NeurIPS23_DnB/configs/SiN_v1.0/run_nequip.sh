#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v1.0/nequip.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp-debug/SiN_v1.0/NequIP
EXPID=normalizer_debug_1gpu

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --seed 123
    #--save-ckpt-every-epoch 1 \

cd $CURRENT_PATH

