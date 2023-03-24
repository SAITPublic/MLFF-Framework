#!/bin/bash

GPU=$1
#MOL=$2
#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml
EXPID=Train950_Rmax5_otf_ReduceLROnPlateau_LR1e-3_EP2000_E1e-3_MAE_F999e-3_L2MAE_EMA999_BS1_1GPU 
EXPHOME=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17


if [ $GPU -eq 0 ]; then

molecules=('aspirin' 'azobenzene' 'benzene') #'ethanol')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/GemNet-T

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 50

done

elif [ $GPU -eq 6 ]; then

molecules=('malonaldehyde' 'naphthalene' 'paracetamol')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/GemNet-T/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-3_EP2000_E1e-3_MAE_F999e-3_L2MAE_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 100

done


elif [ $GPU -eq 7 ]; then

molecules=('salicylic' 'toluene' 'uracil')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/GemNet-T/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-3_EP2000_E1e-3_MAE_F999e-3_L2MAE_EMA999_BS32_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 100

done


fi

cd $CURRENT_PATH
