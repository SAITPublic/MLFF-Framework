#!/bin/bash

CONFIG_PATH_ARRAY=(
  '/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/evaluate_by_metrics/distribution_funcs_HfO_n_super_2.yml'
  '/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/evaluate_by_metrics/distribution_funcs_HfO.yml'
)

CONFIG_PATH_TEMP='/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/evaluate_by_metrics/distribution_funcs_HfO_temp_1.yml'

STRUCTURE_ARRAY=(
    '1200K_crystal_1_5/1'
    '1200K_crystal_1_5/2'
    '1200K_crystal_1_5/3'
    '1200K_crystal_1_5/4'
    '1200K_crystal_1_5/5'
    '1200K_stoichiometry_11_16/11'
    '1200K_stoichiometry_11_16/12'
    '1200K_stoichiometry_11_16/13'
    '1200K_stoichiometry_11_16/14'
    '1200K_stoichiometry_11_16/15'
    '1200K_stoichiometry_11_16/16'
    '1800K_crystal_6_10/6'
    '1800K_crystal_6_10/7'
    '1800K_crystal_6_10/8'
    '1800K_crystal_6_10/9'
    '1800K_crystal_6_10/10'
)
 
for CONFIG_PATH in "${CONFIG_PATH_ARRAY[@]}"
do
  echo ${CONFIG_PATH}
  for struct in "${STRUCTURE_ARRAY[@]}"
  do
    echo ${struct}
    sed "s@{STRUCTURE}@${struct}@g" ${CONFIG_PATH} > ${CONFIG_PATH_TEMP}

    python main.py \
      --mode evaluate \
      --evaluation-metric distribution_functions \
      --evaluation-config-yml ${CONFIG_PATH_TEMP} 

    rm ${CONFIG_PATH_TEMP}
  done
done
