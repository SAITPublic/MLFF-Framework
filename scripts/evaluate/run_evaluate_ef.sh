#!/bin/bash

GPU=$1
CKPT=$2
DATA=$3

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode evaluate \
    --evaluation-metric ef \
    --checkpoint ${CKPT} \
    --reference-trajectory $DATA \
    # --measure-time \
    # --save-ef \
