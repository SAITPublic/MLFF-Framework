#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v1.0/allegro.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v1.0/Allegro
#EXPID=Rmax5_LinearLR_LR5e-3_EP300_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1GPU
EXPID=small_arch5_3L_o3_restr

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --seed 123456
    #--save-ckpt-every-epoch 10 \

cd $CURRENT_PATH

