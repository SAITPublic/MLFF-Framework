#!/bin/bash

GPU=$1
MODEL=$2
CKPT_PATH=$(realpath $3)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG='configs/evaluate/eos.yml'
CONFIG_TEMP='configs/evaluate/eos_temp.yml'

STRUCTURE_ARRAY=(
  'P21c_96atoms'
  'P42nmc_96atoms'
  'Fm3m_96atoms'
  'Pbca_96atoms'
  'Pnma_96atoms'
  'Fd3m_88atoms'
)

for strcut in "${STRUCTURE_ARRAY[@]}"
do
  echo ${strcut}
  sed "s@{STRUCTURE}@${strcut}@g" ${CONFIG} > ${CONFIG_TEMP}
  sed -i "s@{MODEL_NAME}@${MODEL}@g" ${CONFIG_TEMP}
  sed -i "s/{DATA}/HfO/g" ${CONFIG_TEMP}

  CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode evaluate \
    --evaluation-config-yml ${CONFIG_TEMP} \
    --checkpoint ${CKPT_PATH}  \
  
  rm ${CONFIG_TEMP}
done