#!/bin/bash

GPU=$1
MODEL=$2
CKPT_PATH=$(realpath $3)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG='configs/evaluate/eos.yml'
CONFIG_TEMP='configs/evaluate/eos_temp.yml'

STRUCTURE_ARRAY=(
  'P31c_28atoms'
  'P1_14atoms'
  'P1_28atoms'
  'Fd3m_14atoms'
  'Fd3m_56atoms'
)

for strcut in "${STRUCTURE_ARRAY[@]}"
do
  echo ${strcut}
  sed "s@{STRUCTURE}@${strcut}@g" ${CONFIG} > ${CONFIG_TEMP}
  sed -i "s@{MODEL_NAME}@${MODEL}@g" ${CONFIG_TEMP}
  sed -i "s/{DATA}/SiN/g" ${CONFIG_TEMP}

  CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode evaluate \
    --evaluation-config-yml ${CONFIG_TEMP} \
    --checkpoint ${CKPT_PATH}  \
  
  rm ${CONFIG_TEMP}
done