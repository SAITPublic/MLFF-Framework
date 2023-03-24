#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/gemnet-T.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/GemNet-T
EXPID=Rmax5_MaxNeigh50_otf_ReduceLROnPlateau_LR5e-3_EP80_E1_MAE_F100_L2MAE_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 5

cd $CURRENT_PATH

