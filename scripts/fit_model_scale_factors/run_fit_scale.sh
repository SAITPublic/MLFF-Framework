#!/bin/bash

GPU=$1
MODEL=$2
DATA=$3

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG=configs/train/${DATA}/${MODEL}.yml
SCALE_DIR=fit_scale_results/${DATA}/${MODEL}
SCALE_FILE=${MODEL}_scale_factors.json

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode fit-scale \
    --config-yml $CONFIG \
    --scale-path $SCALE_DIR \
    --scale-file $SCALE_FILE \
