#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v1.0/nequip.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v1.0/NequIP
EXPID=Rmax5_LinearLR_LR1e-2_EP300_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS32_4GPU

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=4 /nas/SAIT-MLFF-Framework/main.py \
    --distributed \
    --num-gpus 4 \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10 
    #--seed 123

cd $CURRENT_PATH

