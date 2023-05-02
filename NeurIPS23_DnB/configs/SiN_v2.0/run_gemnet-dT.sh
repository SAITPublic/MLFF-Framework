#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/GemNet-dT

# OCP config
#CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/gemnet-dT.yml
#EXPID=Rmax6_MaxNeigh50_otf_NormOn_ReduceLROnPlateau_LR5e-3_EP80_E1_MAE_F100_L2MAE_EMA999_BS8_1GPU 

# with some SAIT modifications
#EXPID=Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR5e-4_EP200_E1_MAE_F100_L2MAE_EMA999_BS8_1V100 

# SAIT research
## modify both NN arch and training config
#CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/SAIT_research/gemnet-dT-SiN-v1-config.yml
#EXPID=SiN-v1-config_Rmax5_otf_NoNorm_LinearLR_LR1e-2_EP300_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA999_BS8_1GPU 

## modify training config
#CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/SAIT_research/gemnet-dT.yml
#EXPID=Rmax6_MaxNeigh50_otf_LinearLR_LR5e-3_EP80_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA999_BS4_1GPU 


# paper model config
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v2.0/paper_models/gemnet-dT.yml
#EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR5e-4_EP200_E1_MAE_F100_L2MAE_EMA999_BS4_1V100 
#EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOff_LinearLR_LR5e-4_EP200_E1_MAE_F100_L2MAE_EMA999_BS4_1V100 

# norm off + SAIT loss
#EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOff_LinearLR_LR5e-4_EP200_SAITLoss_EMA999_BS4_1V100 

# norm per atom + SAIT loss
EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR5e-4_EP200_SAITLoss_EMA999_BS4_1V100 


CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10

cd $CURRENT_PATH

