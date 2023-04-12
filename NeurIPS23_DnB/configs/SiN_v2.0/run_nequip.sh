#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/NequIP
CONFIG=NeurIPS23_DnB/configs/SiN_v2.0/nequip.yml

# SAIT config
#EXPID=Rmax6_MaxNeigh50_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100

# weight decay
#EXPID=Rmax6_MaxNeigh50_LinearLR_LR5e-3_WD1e-5_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100

# resnet true
EXPID=Rmax6_MaxNeigh50_ResBlock_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
#    --save-ckpt-every-epoch 10

cd $CURRENT_PATH

