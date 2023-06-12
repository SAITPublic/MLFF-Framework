#!/bin/bash

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG='configs/evaluate/distribution_funcs.yml'
CONFIG_TEMP='configs/evaluate/dfs_temp.yml'

STRUCTURE_ARRAY=(
  'monoclinic_1200K'
  'tetragonal_1200K'
  'cubic1_1200K'
  'orthorhombic1_1200K'
  'orthorhombic2_1200K'
  'monoclinic_1800K'
  'tetragonal_1800K'
  'cubic1_1800K'
  'orthorhombic1_1800K'
  'orthorhombic2_1800K'
  'hexagonal1_1200K'
  'hexagonal2_1200K'
  'cubic2_1200K'
)

for struct in "${STRUCTURE_ARRAY[@]}"
do
  echo ${CONFIG}
  echo ${struct}
  # specify a configuration file from the template
  sed "s@{STRUCTURE}@${struct}@g" ${CONFIG} > ${CONFIG_TEMP}
  sed -i "s/{DATA}/HfO/g" ${CONFIG_TEMP}

  python main.py \
    --mode evaluate \
    --evaluation-config-yml ${CONFIG_TEMP} 

  rm ${CONFIG_TEMP}
done
