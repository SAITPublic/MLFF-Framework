#!/bin/bash

GPU=$1
#MOL=$2

#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/dimenet_pp.yml
EXPID=Train950_Rmax5_otf_WarmupConstantLR_LR1e-3_EP3000_E1_MAE_F100_ForcePerDimMAE_EMA999_BS32_1GPU 
EXPHOME=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17

if [ $GPU -eq 3 ]; then

molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/DimeNet++/

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 50

done

elif [ $GPU -eq 4 ]; then

molecules=('malonaldehyde' 'naphthalene' 'paracetamol')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/DimeNet++/

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 50

done


elif [ $GPU -eq 5 ]; then

molecules=('salicylic' 'toluene' 'uracil')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/DimeNet++/

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 50

done


fi

cd $CURRENT_PATH
