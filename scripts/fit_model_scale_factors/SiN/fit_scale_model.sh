#!/bin/bash

GPU=$1
MODEL=$2

BENCHMARK_HOME=/nas/SAIT-MLFF-Framework

if [ $MODEL = "PaiNN" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/SiN_v2.0/painn.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/scales/
SCALEFILE=painn_scale.json

elif [ $MODEL = "GemNet-T" ]; then
#CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/SiN_v1.0/gemnet-T.yml
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/SiN_v2.0/paper_models/gemnet-T.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/scales/
#SCALEFILE=gemnet-T_scale.json
SCALEFILE=gemnet-T_paper_model_scale.json

elif [ $MODEL = "GemNet-dT" ]; then
#CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/SiN_v2.0/gemnet-dT.yml
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/SiN_v2.0/paper_models/gemnet-dT.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/scales/
#SCALEFILE=gemnet-dT_scale.json
SCALEFILE=gemnet-dT_paper_model_scale.json

# employing SiN_v1.0 config
#CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/SiN_v2.0/gemnet-dT-SiN-v1-config.yml
#SCALEFILE=gemnet-dT-SiN-v1-config_scale_with_normalized_labels.json

elif [ $MODEL = "GemNet-OC" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/SiN_v2.0/gemnet-OC.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/SiN_v2.0/scales/
SCALEFILE=gemnet-OC_scale.json

fi

CURRENT_PATH=${pwd}
cd $BENCHMARK_HOME

CUDA_VISIBLE_DEVICES=$GPU python ${BENCHMARK_HOME}/main.py \
    --mode fit-scale \
    --config-yml $CONFIG \
    --scale-path $SCALEPATH \
    --scale-file $SCALEFILE \
    --num-batches 64

cd $CURRENT_PATH
