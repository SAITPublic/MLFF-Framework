#!/bin/bash

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG='configs/evaluate/distribution_funcs.yml'
CONFIG_TEMP='configs/evaluate/dfs_temp.yml'

STRUCTURE_ARRAY=(
  'amorphous_1200K'
  'triclinic_1200K'
  'cubic1_1200K'
  'cubic2_1200K'
)

for struct in "${STRUCTURE_ARRAY[@]}"
do
  echo ${CONFIG}
  echo ${struct}
  # specify a configuration file from the template
  sed "s@{STRUCTURE}@${struct}@g" ${CONFIG} > ${CONFIG_TEMP}
  sed -i "s/{DATA}/SiN/g" ${CONFIG_TEMP}

  python main.py \
    --mode evaluate \
    --evaluation-config-yml ${CONFIG_TEMP} 

  rm ${CONFIG_TEMP}
done
