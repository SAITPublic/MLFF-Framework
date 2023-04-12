#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=$(realpath ../../)

cd $BENCHMARK_HOME

CKPT_DIR=$2
DATA_PATH=$3

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_PATH \
#    --validate-batch-size 1


cd $CURRENT_PATH
