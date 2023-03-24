#!/bin/bash

GPU=$1
MODEL=$2
MOL=$3

BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

if [ $MODEL = "PaiNN" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/rMD17/painn.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/scales/
SCALEFILE=painn_${MOL}_scale.json

elif [ $MODEL = "GemNet-T" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/scales/
SCALEFILE=gemnet-T_${MOL}_scale.json

elif [ $MODEL = "GemNet-dT" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/rMD17/gemnet-dT.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/scales/
SCALEFILE=gemnet-dT_${MOL}_scale.json

elif [ $MODEL = "GemNet-OC" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/rMD17/gemnet-OC.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/rMD17/scales/
SCALEFILE=gemnet-OC_${MOL}_scale.json

fi

CURRENT_PATH=${pwd}
cd $BENCHMARK_HOME

CUDA_VISIBLE_DEVICES=$GPU python ${BENCHMARK_HOME}/main.py \
    --mode fit-scale \
    --config-yml $CONFIG \
    --molecule $MOL \
    --scale-path $SCALEPATH \
    --scale-file $SCALEFILE

cd $CURRENT_PATH