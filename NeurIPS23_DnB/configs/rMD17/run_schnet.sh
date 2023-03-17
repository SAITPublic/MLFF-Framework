#!/bin/bash

GPU=$1
#MOL=$2

#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

MOL='benzene'
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/schnet.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp-debug/rMD17/${MOL}/SchNet
EXPID=Train1K_Rmax5_LinearLR_LR1e3_EP3200_E1e-2_MAE_F99e-2_MAE_EMA99_BS32_1GPU

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 100

cd $CURRENT_PATH


if [ $GPU -eq 4 ]; then

molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/schnet.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp-debug/rMD17/${MOL}/SchNet/
EXPID=Train1K_Rmax5_LinearLR_LR1e3_EP3200_E1e-2_MAE_F99e-2_MAE_EMA99_BS32_1GPU

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done


elif [ $GPU -eq 5 ]; then

molecules=('naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

for MOL in "${molecules[@]}"; do


CONFIG=/nas/NeurIPS23_DnB/configs/rMD17/schnet.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/SchNet/
EXPID=Train1K_Rmax5_LinearLR_LR1e3_EP3200_E1e-2_MAE_F99e-2_MAE_EMA99_BS32_1GPU

CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done

fi
