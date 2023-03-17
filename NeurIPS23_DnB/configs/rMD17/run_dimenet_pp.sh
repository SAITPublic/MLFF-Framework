#!/bin/bash

#BASEDIR=/home/workspace/MLFF/
#CONFIG=${BASEDIR}/configs/OC20-2M/schnet.yml
#EXPDIR=${BASEDIR}/exp/OC20-2M/SchNet/

GPU=$1
MOL=$2

#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

if [ $GPU -eq 1 ]; then

molecules=('azobenzene' 'benzene' 'ethanol')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/NeurIPS23_DnB/configs/rMD17/dimenet_pp.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/DimeNet++/
EXPID=Train1K_Rmax5_WarmupLinearLR_LR1e-3_EP3200_E1_MAE_F100_MAEPerDim_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done

elif [ $GPU -eq 2 ]; then

molecules=('malonaldehyde' 'naphthalene' 'paracetamol')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/NeurIPS23_DnB/configs/rMD17/dimenet_pp.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/DimeNet++/
EXPID=Train1K_Rmax5_WarmupLinearLR_LR1e-3_EP3200_E1_MAE_F100_MAEPerDim_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done


elif [ $GPU -eq 3 ]; then

molecules=('salicylic' 'toluene' 'uracil')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/NeurIPS23_DnB/configs/rMD17/dimenet_pp.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/DimeNet++/
EXPID=Train1K_Rmax5_WarmupLinearLR_LR1e-3_EP3200_E1_MAE_F100_MAEPerDim_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done


fi
