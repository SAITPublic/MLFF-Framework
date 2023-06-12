#!/bin/bash

GPU=$1
MODEL=$2
CKPT_PATH=$(realpath $3)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG='configs/simulate/md_sim_config.yml'
CONFIG_TEMP='configs/simulate/md_sim_temp.yml'

STRUCTURE_ARRAY=(
  'amorphous_1200K'
  'triclinic_1200K'
  'cubic1_1200K'
  'cubic2_1200K'
)

for supercell in 1 2 3
do
  for struct in "${STRUCTURE_ARRAY[@]}"
  do
    # specify a configuration file from the template
    sed "s@{STRUCTURE}@${struct}@g" ${CONFIG} > ${CONFIG_TEMP}
    sed -i "s@{MODEL}@${MODEL}@g" ${CONFIG_TEMP}
    sed -i "s/{TEMP_K}/1200/g" ${CONFIG_TEMP}
    sed -i "s/{DATA}/SiN/g" ${CONFIG_TEMP}

    echo ${CONFIG}
    if [ $supercell == 1 ]; then
      echo ${struct} 
    elif [ $supercell == 2 ]; then
      echo ${struct} with supercell 2x2x2
      echo "n_super: [2,2,2]" >> ${CONFIG_TEMP}
    elif [ $supercell == 3 ]; then
      echo ${struct} with supercell 3x3x3
      echo "n_super: [3,3,3]" >> ${CONFIG_TEMP}
    fi

    CUDA_VISIBLE_DEVICES=$GPU python main.py \
      --mode run-md \
      --md-config-yml ${CONFIG_TEMP} \
      --checkpoint ${CKPT_PATH} \
    
    rm ${CONFIG_TEMP}
  done
done