#!/bin/bash

MODEL=$1

GPU=$2

MOL=$3

CKPT=$4

echo "Generate a scale file of $MODEL"

if [ $MODEL = "PaiNN" ]; then
## using dummy mode!
CUDA_VISIBLE_DEVICES=$GPU python ocpmodels/modules/scaling/fit.py \

elif [ $MODEL = "GemNet-T" ]; then
## using dummy mode!
CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/ocpmodels/modules/scaling/fit.py \
    --mode prepare-checkpoint \
    --config-yml /nas/NeurIPS23_DnB/configs/rMD17/gemnet-T.yml \
    --run-dir /home/workspace/MLFF/tmp/rMD17/GemNet-T \
    --identifier scale_file_${MOL} \
    --molecule $MOL \
    --output-scale-file scale_file_gemnet-T_${MOL}.json \
    --checkpoint $CKPT


elif [ $MODEL = "GemNet-dT" ]; then
## using dummy mode!
CUDA_VISIBLE_DEVICES=$GPU python /nas/ocp/ocpmodels/modules/scaling/fit.py \
    --mode prepare-checkpoint \
    --config-yml /nas/NeurIPS23_DnB/configs/rMD17/gemnet-dT.yml \
    --run-dir /home/workspace/MLFF/tmp/rMD17/GemNet-dT \
    --identifier scale_file_${MOL} \
    --molecule $MOL \
    --output-scale-file scale_file_${MOL}.json \
    --checkpoint $CKPT

elif [ $MODEL = "GemNet-OC" ]; then
## using dummy mode!
CUDA_VISIBLE_DEVICES=$GPU python ocpmodels/modules/scaling/fit.py \


fi

