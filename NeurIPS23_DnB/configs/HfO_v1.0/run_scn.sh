#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/HfO_v1.0/dataset_2/SCN

# OCP config
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/HfO_v1.0/scn.yml
#EXPID=Rmax6_MaxNeigh50_otf_NormOn_ReduceLROnPlateau_LR5e-3_EP80_E1_MAE_F100_L2MAE_EMA999_BS4_1V100

# with some SAIT modifications
#EXPID=Rmax8_MaxNeigh40_otf_NormOn_LinearLR_LR4e-4_EP200_E2_MAE_F100_L2MAE_EMA999_BS3_1V100 
EXPID=Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR4e-4_EP200_E2_MAE_F100_L2MAE_EMA999_BS4_2V100 


CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10

cd $CURRENT_PATH

