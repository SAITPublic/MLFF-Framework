#!/bin/bash

GPU=0

CONFIG_PATH='/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/evaluate_by_metrics/pe_well.yml'
CONFIG_PATH_TEMP='/workspace/packages/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/evaluate_by_metrics/pe_well_temp.yml'

MODELS=(
  # 'Allegro/Rmax6_MaxNeigh50_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100-20230408_072648'
  # 'BPNN/Rmax6_MaxNeigh50_NormPerAtomOn_LinearLR_LR5e-3_EP200_SAITLoss_BS16_1V100-20230502_075925'
  # 'DimeNet++/Paper_Model_Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR5e-3_EP200_SAITLoss_EMA999_BS8_1V100-20230426_071513'
  # 'GemNet-T/Paper_Model_Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR5e-4_EP200_SAITLoss_EMA999_BS8_1V100-20230426_071146'
  # 'GemNet-dT/Paper_Model_Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR5e-3_EP200_SAITLoss_EMA999_BS8_1V100-20230426_071855'
  # 'MACE/Rmax6_MaxNeigh50_LinearLR_LR1e-2_EP200_AMSGradOff_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100-20230424_235240'
  # 'NequIP/Rmax6_MaxNeigh50_LinearLR_LR5e-3_EP200_E1_EnergyPerAtomMSE_F1_ForcePerDimMSE_EMA99_BS16_1V100-20230407_065101'
  # 'SchNet/Rmax6_MaxNeigh50_otf_NormPerAtomOn_LinearLR_LR1e-4_EP200_SAITLoss_BS16_1V100-20230428_040751'
  'GemNet-T/Paper_Model_Rmax6_MaxNeigh50_otf_NormOff_LinearLR_LR5e-4_EP200_SAITLoss_EMA999_BS8_1V100-20230426_070942'
)
 
for MODEL in "${MODELS[@]}"
do
  echo ${MODEL}
  MODEL_STR_ARRAY=(`echo $MODEL | tr "/" "\n"`)
  echo ${MODEL_STR_ARRAY[0]}

  sed "s@{MODEL}@${MODEL}@g" ${CONFIG_PATH} > ${CONFIG_PATH_TEMP}

  CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode evaluate \
    --evaluation-metric pe_well \
    --evaluation-config-yml ${CONFIG_PATH_TEMP} \
    --checkpoint /workspace/for_benchmark/HfO_v1/models/${MODEL}/checkpoint.pt  \
  
  rm ${CONFIG_PATH_TEMP}

done