#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v1.0/MACE

# SAIT config (SiN v1 default)
CONFIG=NeurIPS23_DnB/configs/SiN_v1.0/mace.yml
EXPID=SiNv1_config_Rmax5_LinearLR_LR1e-2_EP300_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS32_1V100

# single GPU
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    #--save-ckpt-every-epoch 10

cd $CURRENT_PATH

