#!/bin/bash

GPU=$1

CURRENT_PATH=${pwd}
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/GemNet-T

# OCP config 
CONFIG=NeurIPS23_DnB/configs/SiN_v2.0/gemnet-T.yml
#EXPID=Rmax5_MaxNeigh50_otf_ReduceLROnPlateau_LR5e-3_EP80_E1_MAE_F100_L2MAE_EMA999_BS32_1GPU 

# with some SAIT modifications
EXPID=Rmax6_MaxNeigh50_otf_LinearLR_LR5e-4_EP200_E1_MAE_F100_L2MAE_EMA999_BS4_4GPU 



# paper model config
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/SiN_v1.0/paper_models/gemnet-T.yml
#EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR5e-4_EP200_E1_MAE_F100_L2MAE_EMA999_BS4_1V100 
EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOff_LinearLR_LR5e-4_EP200_E1_MAE_F100_L2MAE_EMA999_BS4_1V100 



# single GPU
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
#    --save-ckpt-every-epoch 10

# 4 GPU
#CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=4 main.py \
#    --distributed \
#    --num-gpus 4 \
#    --mode train \
#    --config-yml $CONFIG \
#    --run-dir $EXPDIR \
#    --identifier $EXPID \
#    --print-every 100 \
#    --save-ckpt-every-epoch 10 \
#    --checkpoint $2 


cd $CURRENT_PATH

