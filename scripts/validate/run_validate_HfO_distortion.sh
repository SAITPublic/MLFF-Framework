#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=$(realpath ../../)

cd $BENCHMARK_HOME

CKPT_DIR=$2
DATA_PATH=/home/workspace/MLFF/HfO_v1.0/symmetric_distortion/atom_graph_rmax6.0_maxneighbor50

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_PATH/test_0.01.lmdb \
    --validate-batch-size 16

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_PATH/test_0.03.lmdb \
    --validate-batch-size 16

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_PATH/test_0.09.lmdb \
    --validate-batch-size 16

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_PATH/test_0.27.lmdb \
    --validate-batch-size 16

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_PATH/test_0.81.lmdb \
    --validate-batch-size 16

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_PATH/test.lmdb \
    --validate-batch-size 16


cd $CURRENT_PATH
