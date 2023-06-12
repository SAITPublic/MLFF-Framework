#!/bin/bash

GPU=$1
MODEL=$2
CKPT_PATH=$(realpath $3)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG='configs/evaluate/pec.yml'
CONFIG_TEMP='configs/evaluate/pec_temp.yml'

s1=('SiSi' 'interatomic_distances' 1.0 8.0 0.1 'energy_distance.dat')
s2=('SiN' 'interatomic_distances' 1.0 8.0 0.1 'energy_distance.dat')
s3=('NN' 'interatomic_distances' 0.5 8.0 0.1 'energy_distance.dat')
s4=('Si3N' 'filename_scale_suffixes' 2.0 6.0 0.1 'energy_distance.dat')
s5=('SiN4' 'filename_scale_suffixes' 2.0 6.0 0.1 'energy_distance.dat')
STRUCTURE_ARRAY=(
  s1[@]
  s2[@]
  s3[@]
  s4[@]
  s5[@]
)

COUNT=${#STRUCTURE_ARRAY[@]}
for (( index=0; index<$COUNT; index++ ))
do
  struct=${!STRUCTURE_ARRAY[index]:0:1}
  pec_param=${!STRUCTURE_ARRAY[index]:1:1}
  start=${!STRUCTURE_ARRAY[index]:2:1}
  end=${!STRUCTURE_ARRAY[index]:3:1}
  interval=${!STRUCTURE_ARRAY[index]:4:1}
  energy_relation=${!STRUCTURE_ARRAY[index]:5:1}

  echo ${struct}  
  sed "s@{STRUCTURE}@${struct}@g" ${CONFIG} > ${CONFIG_TEMP}
  sed -i "s@{MODEL}@${MODEL}@g" ${CONFIG_TEMP}
  sed -i "s/{DATA}/SiN/g" ${CONFIG_TEMP}
  sed -i "s@{PEC_PARAM}@${pec_param}@g" ${CONFIG_TEMP}
  sed -i "s@{START}@${start}@g" ${CONFIG_TEMP}
  sed -i "s@{END}@${end}@g" ${CONFIG_TEMP}
  sed -i "s@{INTERVAL}@${interval}@g" ${CONFIG_TEMP}
  sed -i "s@{ENERGY_RELATION}@${energy_relation}@g" ${CONFIG_TEMP}

  CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode evaluate \
    --evaluation-config-yml ${CONFIG_TEMP} \
    --checkpoint ${CKPT_PATH} \

  rm ${CONFIG_TEMP}

done