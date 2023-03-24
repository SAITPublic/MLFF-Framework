#!/bin/bash

GPU=$1
#MOL=$2

#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/scn.yml
EXPID=Train950_Rmax5_otf_ReduceLROnPlateau_LR1e-3_EP2000_E1e-2_MSE_F99e-2_ForcePerDimMSE_EMA99_BS10_1GPU 
EXPHOME=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17


if [ $GPU -eq 1 ]; then

molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'uracil')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/SCN

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 50

done

elif [ $GPU -eq 2 ]; then

molecules=('malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene')
for MOL in "${molecules[@]}"; do
EXPDIR=${EXPHOME}/${MOL}/SCN

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
