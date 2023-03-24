#!/bin/bash

GPU=$1
#MOL=$2

#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/schnet.yml
EXPID=Train950_Rmax5_otf_ConstantLR_LR1e-3_EP3000_E1e-2_MSE_F1_L2MAE_EMA99_BS32_1GPU
EXPHOME=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17

if [ $GPU -eq 6 ]; then

molecules=('aspirin' 'azobenzene' 'naphthalene' 'ethanol' 'malonaldehyde')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/SchNet

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 100

done


elif [ $GPU -eq 7 ]; then

molecules=('benzene' 'paracetamol' 'salicylic' 'toluene' 'uracil')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/SchNet

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
