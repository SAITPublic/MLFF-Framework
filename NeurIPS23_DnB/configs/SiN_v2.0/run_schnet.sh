#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/SchNet

# OCP config
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/schnet.yml
#EXPID=Rmax6_MaxNeigh50_otf_WarmupStepLR_LR1e-4_EP80_E1_MSE_F100_MSE_BS16_1GPU

# with some SAIT modifications
#EXPID=Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR1e-4_EP200_E1_MSE_F100_MSE_BS16_1V100
EXPID=Rmax6_MaxNeigh50_NormOn_LinearLR_LR1e-4_EP200_E1_MSE_F100_MSE_BS16_1V100



EXPID=Rmax6_MaxNeigh50_NormOff_LinearLR_LR1e-4_EP200_E1_MSE_F100_MSE_BS16_1V100

# SAIT research
#CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/SAIT_research/schnet.yml
#EXPID=Rmax6_MaxNeigh50_otf_LinearLR_LR1e-4_EP80_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_BS16_1GPU 
#EXPID=Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR1e-4_EP80_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_BS16_1GPU 
#EXPID=Rmax6_MaxNeigh50_otf_LinearLR_LR1e-4_EP80_E1_MSE_F100_MSE_BS16_1GPU 


CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10

cd $CURRENT_PATH

