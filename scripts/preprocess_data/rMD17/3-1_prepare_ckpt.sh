#!/bin/bash

MODEL=$1

GPU=$2

MOL=$3

echo "Prepare checkpoint to generate scale file - $MODEL"

if [ $MODEL = "PaiNN" ]; then
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode prepare-checkpoint \

elif [ $MODEL = "GemNet-T" ]; then
CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode prepare-checkpoint \
    --config-yml /nas/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml \
    --run-dir /home/workspace/MLFF/tmp/rMD17/GemNet-T \
    --identifier scale_file_${MOL} \
    --molecule $MOL

elif [ $MODEL = "GemNet-dT" ]; then
CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/main.py \
    --mode prepare-checkpoint \
    --config-yml /nas/NeurIPS23_DnB/configs/rMD17/gemnet-dT.yml \
    --run-dir /home/workspace/MLFF/tmp/rMD17/GemNet-dT \
    --identifier scale_file_${MOL} \
    --molecule $MOL

elif [ $MODEL = "GemNet-OC" ]; then
CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode prepare-checkpoint \


fi

