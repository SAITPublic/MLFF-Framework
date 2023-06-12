#!/bin/bash

GPU=$1
CKPT_DIR=$(realpath $2)
DATA_LMDB=$(realpath $3)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_LMDB \
#    --validate-batch-size 16 \
