#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/HfO_v1.0/dataset_3/NequIP

# SAIT config
CONFIG=NeurIPS23_DnB/configs/HfO_v1.0/nequip.yml
#EXPID=Rmax6_MaxNeigh50_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100

# norm off + SAIT loss (same in NequIP loss)
EXPID=Rmax6_MaxNeigh50_NormOff_LinearLR_LR5e-3_EP200_SAITLoss_EMA99_BS16_1V100


# single GPU
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10

cd $CURRENT_PATH

