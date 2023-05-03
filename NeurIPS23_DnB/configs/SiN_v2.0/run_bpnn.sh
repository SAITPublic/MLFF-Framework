#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/BPNN

# SAIT config
CONFIG=NeurIPS23_DnB/configs/SiN_v2.0/bpnn.yml
#EXPID=Rmax6_MaxNeigh50_NormOn_LinearLR_LR1e-4_EP200_E1_EnergyPerAtomMSE_F1e-1_ForcePerDimMSE_BS16_1V100


#EXPID=Rmax6_MaxNeigh50_NormOff_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1e-1_ForcePerDimMSE_BS16_1V100

# norm off + SAIT loss
EXPID=Rmax6_MaxNeigh50_NormOff_LinearLR_LR5e-3_EP200_SAITLoss_BS16_1V100 

# norm per atom + SAIT loss
#EXPID=Rmax6_MaxNeigh50_NormPerAtomOn_LinearLR_LR5e-3_EP200_SAITLoss_BS16_1V100 

# norm per snapshot + SAIT loss
#EXPID=Rmax6_MaxNeigh50_NormOn_LinearLR_LR5e-3_EP200_SAITLoss_BS16_1V100 

# single GPU
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10 

cd $CURRENT_PATH

