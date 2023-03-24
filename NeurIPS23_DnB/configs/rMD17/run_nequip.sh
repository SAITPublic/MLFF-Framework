#!/bin/bash

GPU=$1
#MOL=$2
#molecules=('aspirin' 'azobenzene' 'benzene' 'ethanol' 'malonaldehyde' 'naphthalene' 'paracetamol' 'salicylic' 'toluene' 'uracil')

CURRENT_PATH=${pwd}
BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

cd $BENCHMARK_HOME

if [ $GPU -eq 4 ]; then

molecules=('azobenzene')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/nequip.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/NequIP/
EXPID=Train950_Rmax5_ReduceLROnPlateau_LR1e-2_EP2000_E1_MSE_F1000_ForcePerDimMSE_EMA99_BS5_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 50

done


elif [ $GPU -eq 5 ]; then

molecules=('benzene' 'ethanol')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/nequip.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/NequIP/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-2_EP2000_E1_EnergyPerAtomMAE_F1000_ForcePerDimMAE_EMA99_BS5_1GPU 

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

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/nequip.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/NequIP/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-2_EP2000_E1_EnergyPerAtomMAE_F1000_ForcePerDimMAE_EMA99_BS5_1GPU 

CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --molecule $MOL \
    --save-ckpt-every-epoch 50

done


elif [ $GPU -eq 7 ]; then

molecules=('salicylic' 'toluene' 'uracil')

for MOL in "${molecules[@]}"; do

CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/rMD17/nequip.yml
EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/${MOL}/NequIP/
EXPID=Train1K_Rmax5_ReduceLROnPlateau_LR1e-2_EP2000_E1_EnergyPerAtomMAE_F1000_ForcePerDimMAE_EMA99_BS5_1GPU 

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
