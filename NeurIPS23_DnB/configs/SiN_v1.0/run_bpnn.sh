#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v1.0/bpnn.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v1.0/BPNN
EXPID=Rmax6_ConstantLR_LR1e-4_EP1000_E1_EnergyPerAtomMSE_F1e-1_ForcePerDimMSE_BS8_1GPU

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --seed 123
    #--save-ckpt-every-epoch 1 \

cd $CURRENT_PATH

