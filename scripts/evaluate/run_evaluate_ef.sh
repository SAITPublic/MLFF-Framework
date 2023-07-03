#!/bin/bash

GPU=$1
CKPT=$(realpath $2)
DATA=$(realpath $3)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode evaluate \
    --evaluation-metric ef \
    --checkpoint ${CKPT} \
    --reference-trajectory $DATA \
    # --save-ef \
    # --measure-time \