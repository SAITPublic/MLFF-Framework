#!/bin/bash

GPU=$1
MODEL=$2

BENCHMARK_HOME=$(realpath ../../../)

if [ $MODEL = "PaiNN" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/HfO_v1.0/painn.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/HfO_v1.0/scales/
SCALEFILE=painn_scale_dataset_2.json

elif [ $MODEL = "GemNet-T" ]; then
#CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/HfO_v1.0/gemnet-T.yml
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/HfO_v1.0/paper_models/gemnet-T.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/HfO_v1.0/scales/
#SCALEFILE=gemnet-T_scale_dataset_2.json
SCALEFILE=gemnet-T_paper_model_scale_dataset_2.json

elif [ $MODEL = "GemNet-dT" ]; then
#CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/HfO_v1.0/gemnet-dT.yml
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/HfO_v1.0/paper_models/gemnet-dT.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/HfO_v1.0/scales/
#SCALEFILE=gemnet-dT_scale_dataset_2.json
SCALEFILE=gemnet-dT_paper_model_scale_dataset_2.json

# employing SiN_v1.0 config
#CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/HfO_v1.0/gemnet-dT-SiN-v1-config.yml
#SCALEFILE=gemnet-dT-SiN-v1-config_scale_with_normalized_labels.json

elif [ $MODEL = "GemNet-OC" ]; then
CONFIG=${BENCHMARK_HOME}/NeurIPS23_DnB/configs/HfO_v1.0/gemnet-OC.yml
SCALEPATH=/home/workspace/MLFF/NeurIPS23_DnB-exp/HfO_v1.0/scales/
SCALEFILE=gemnet-OC_scale_dataset_2.json

fi


CURRENT_PATH=${pwd}
cd $BENCHMARK_HOME

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode fit-scale \
    --config-yml $CONFIG \
    --scale-path $SCALEPATH \
    --scale-file $SCALEFILE

cd $CURRENT_PATH
