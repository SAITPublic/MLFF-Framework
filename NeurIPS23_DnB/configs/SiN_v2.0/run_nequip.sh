#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/nequip.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/NequIP
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp-debug/ckpt_check
#EXPID=Rmax5_ReduceLROnPlateau_LR5e-3_EP1000_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS1_1GPU 
#EXPID=Rmax5_MaxNeigh50_ReduceLROnPlateau_LR5e-3_EP1000_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS1_1GPU 


# for SAIT config (https://confluence.samsungds.net/display/ESC/Training+with+various+Hyperparameters)
# config 4 + 85 epoch for 22510 snapshots (SiN_v1.0 : 300 epoch for 6402 snapshots) : same num of training samples
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/nequip_config4.yml
#EXPID=Rmax5_LinearLR_LR1e-2_EP85_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS32_1GPU

# config 4 + 300 epoch for 22510 snapshots : same epoch
EXPID=Rmax5_LinearLR_LR1e-2_EP300_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS32_1GPU

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 20

cd $CURRENT_PATH

