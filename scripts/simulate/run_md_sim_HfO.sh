#!/bin/bash

GPU=$1
MODEL=$2
CKPT_PATH=$(realpath $3)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG='configs/simulate/md_sim_config.yml'
CONFIG_TEMP='configs/simulate/md_sim_temp.yml'

STRUCTURE_ARRAY_1200=(
  'monoclinic_1200K'
  'tetragonal_1200K'
  'cubic1_1200K'
  'orthorhombic1_1200K'
  'orthorhombic2_1200K'
  'hexagonal1_1200K'
  'hexagonal2_1200K'
  'cubic2_1200K'
)

STRUCTURE_ARRAY_1800=(
  'monoclinic_1800K'
  'tetragonal_1800K'
  'cubic1_1800K'
  'orthorhombic1_1800K'
  'orthorhombic2_1800K'
)

for supercell in 1 2 3
do
  for struct in "${STRUCTURE_ARRAY_1200[@]}"
  do
    # specify a configuration file from the template
    sed "s@{STRUCTURE}@${struct}@g" ${CONFIG} > ${CONFIG_TEMP}
    sed -i "s@{MODEL}@${MODEL}@g" ${CONFIG_TEMP}
    sed -i "s/{TEMP_K}/1200/g" ${CONFIG_TEMP}
    sed -i "s/{DATA}/HfO/g" ${CONFIG_TEMP}

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

  for struct in "${STRUCTURE_ARRAY_1800[@]}"
  do
    # specify a configuration file from the template
    sed "s@{STRUCTURE}@${struct}@g" ${CONFIG} > ${CONFIG_TEMP}
    sed -i "s@{MODEL}@${MODEL}@g" ${CONFIG_TEMP}
    sed -i "s/{TEMP_K}/1800/g" ${CONFIG_TEMP}
    sed -i "s/{DATA}/HfO/g" ${CONFIG_TEMP}

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
