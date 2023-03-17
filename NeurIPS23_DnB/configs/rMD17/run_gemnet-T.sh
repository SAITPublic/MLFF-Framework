#!/bin/bash

#BASEDIR=/home/workspace/MLFF/
#CONFIG=${BASEDIR}/configs/OC20-2M/schnet.yml
#EXPDIR=${BASEDIR}/exp/OC20-2M/SchNet/

GPU=$1

#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

if [ $GPU -eq 5 ]; then

molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/GemNet-T/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-3_EP2000_E1e-3_MAE_F999e-3_L2MAE_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done

elif [ $GPU -eq 6 ]; then

molecules=('malonaldehyde' 'naphthalene' 'paracetamol')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/GemNet-T/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-3_EP2000_E1e-3_MAE_F999e-3_L2MAE_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done


elif [ $GPU -eq 7 ]; then

molecules=('salicylic' 'toluene' 'uracil')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/GemNet-T/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-3_EP2000_E1e-3_MAE_F999e-3_L2MAE_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-epoch 100

done


fi
