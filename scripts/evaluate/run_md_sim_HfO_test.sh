#!/bin/bash

GPU=3

CONFIG_PATH_ARRAY=(
  '/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/simulate/HfO_2.yml'
  '/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/simulate/HfO.yml'
)
CONFIG_PATH_TEMP='/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/simulate/HfO_temp.yml'

MODELS=(
  'Allegro/Rmax6_MaxNeigh50_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100-20230408_072648'
  # 'BPNN/Rmax6_MaxNeigh50_NormPerAtomOn_LinearLR_LR5e-3_EP200_SAITLoss_BS16_1V100-20230502_075925'
  # 'DimeNet++/Paper_Model_Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR5e-3_EP200_SAITLoss_EMA999_BS8_1V100-20230426_071513'
  # 'GemNet-T/Paper_Model_Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR5e-4_EP200_SAITLoss_EMA999_BS8_1V100-20230426_071146'
  # 'GemNet-dT/Paper_Model_Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR5e-3_EP200_SAITLoss_EMA999_BS8_1V100-20230426_071855'
  # 'MACE/Rmax6_MaxNeigh50_LinearLR_LR1e-2_EP200_AMSGradOff_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100-20230424_235240'
  # 'MACE/Rmax6_MaxNeigh50_LinearLR_LR1e-2_EP200_AMSGradOff_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100-20230412_050734'
  # 'NequIP/Rmax6_MaxNeigh50_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100-20230407_065101'
  # 'SchNet/Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR1e-4_EP200_SAITLoss_BS16_1V100-20230428_040751'
  # 'MACE/Rmax6_MaxNeigh50_LinearLR_LR1e-2_EP200_AMSGradOff_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100_Seed2023-20230516_013407'
)


STRUCTURE_ARRAY_1200=(
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
)

STRUCTURE_ARRAY_1800=(
    '1800K_crystal_6_10/6'
    '1800K_crystal_6_10/7'
    '1800K_crystal_6_10/8'
    '1800K_crystal_6_10/9'
    '1800K_crystal_6_10/10'
)

for CONFIG_PATH in "${CONFIG_PATH_ARRAY[@]}"
do
  echo ${CONFIG_PATH}
  for MODEL in "${MODELS[@]}"
  do
    echo ${CONFIG_PATH}
    echo ${MODEL}

    for struct in "${STRUCTURE_ARRAY_1200[@]}"
    do
      echo ${CONFIG_PATH}
      echo ${MODEL}
      echo ${struct}
      sed "s@{STRUCTURE}@${struct}@g" ${CONFIG_PATH} > ${CONFIG_PATH_TEMP}
      sed -i "s@{MODEL}@${MODEL}@g" ${CONFIG_PATH_TEMP}
      sed -i "s/{TEMP_K}/1200/g" ${CONFIG_PATH_TEMP}

      CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --mode run-md \
        --md-config-yml ${CONFIG_PATH_TEMP} \
        --checkpoint /workspace/for_benchmark/HfO_v1/models/${MODEL}/checkpoint.pt  \
      
      rm ${CONFIG_PATH_TEMP}
    done

    for struct in "${STRUCTURE_ARRAY_1800[@]}"
    do
      echo ${CONFIG_PATH}
      echo ${struct}
      sed "s@{STRUCTURE}@"${struct}"@g" ${CONFIG_PATH} > ${CONFIG_PATH_TEMP}
      sed -i "s@{MODEL}@${MODEL}@g" ${CONFIG_PATH_TEMP}
      sed -i "s/{TEMP_K}/1800/g" ${CONFIG_PATH_TEMP}

      CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --mode run-md \
        --md-config-yml /${CONFIG_PATH_TEMP} \
        --checkpoint /workspace/for_benchmark/HfO_v1/models/${MODEL}/checkpoint.pt
      
      rm ${CONFIG_PATH_TEMP}
    done
    
  done
done