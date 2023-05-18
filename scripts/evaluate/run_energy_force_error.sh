#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=$(realpath ../../)

cd $BENCHMARK_HOME

CKPT=$2
DATA=$3

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode evaluate \
    --evaluation-metric ef \
    --checkpoint ${CKPT} \
    --reference-trajectory $DATA \
    --measure-time-per-snapshot \
#    --save-ef \

