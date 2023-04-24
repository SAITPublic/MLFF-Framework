#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/DimeNet++

# OCP config
CONFIG=NeurIPS23_DnB/configs/SiN_v2.0/dimenet_pp.yml
#EXPID=Rmax6_MaxNeigh50_otf_WarmupStepLR_LR1e-4_EP80_E1_MSE_F50_MSE_BS4_2GPU

# with some SAIT modifications
#EXPID=Rmax6_MaxNeigh50_otf_NormOn_WarmupEP5_LinearLR_LR1e-4_EP200_E1_MSE_F50_MSE_BS4_2GPU
EXPID=Rmax6_MaxNeigh50_NormOn_WarmupEP5_LinearLR_LR1e-4_EP200_E1_MSE_F50_MSE_BS4_2GPU



# paper model config
CONFIG=NeurIPS23_DnB/configs/SiN_v2.0/paper_models/dimenet_pp.yml
#EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR1e-4_EP200_E1_MSE_F50_MSE_BS3_1V100
EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOff_LinearLR_LR1e-4_EP200_E1_MSE_F50_MSE_BS3_1V100


# single GPU
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --seed 1
#    --save-ckpt-every-epoch 10

# 2 GPU
#CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=2 main.py \
#    --distributed \
#    --num-gpus 2 \
#    --mode train \
#    --config-yml $CONFIG \
#    --run-dir $EXPDIR \
#    --identifier $EXPID \
#    --print-every 100 \
#    --save-ckpt-every-epoch 10 \
 

cd $CURRENT_PATH

