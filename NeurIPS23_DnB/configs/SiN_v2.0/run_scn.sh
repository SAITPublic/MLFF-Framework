#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/SCN

# SAIT config
CONFIG=NeurIPS23_DnB/configs/SiN_v2.0/scn.yml
EXPID=Rmax6_MaxNeigh50_NormOn_LinearLR_LR4e-4_EP200_E2_MAE_F100_L2MAE_BS4_4V100

# single GPU
#CUDA_VISIBLE_DEVICES=$GPU python main.py \
#    --mode train \
#    --config-yml $CONFIG \
#    --run-dir $EXPDIR \
#    --identifier $EXPID \
#    --print-every 100 \
#    --save-ckpt-every-epoch 10 


# 2 GPU
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --distributed \
    --num-gpus 4 \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10 \


cd $CURRENT_PATH

