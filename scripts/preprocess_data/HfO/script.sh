#!/bin/bash

DATADIR=/DB/HfO_v1.0
DATASETS=('dataset_1' 'dataset_2' 'dataset_3' 'dataset_4')
# dataset_1 : Crystal
# dataset_2 : Random
# dataset_3 : C+R with fixed split
# dataset_4 : C+R with random split

for DATASET in "${DATASETS[@]}"; do

python preprocess.py \
    --train-data ${DATADIR}/${DATASET}/train.extxyz \
    --valid-data ${DATADIR}/${DATASET}/valid.extxyz \
    --test-data ${DATADIR}/${DATASET}/test.extxyz \
    --out-path /home/workspace/MLFF/HfO_v1.0/${DATASET} \
    --r-max 6.0 \
    --max-neighbors 50 

done

# OOD data
python preprocess.py \
    --test-data ${DATADIR}/${DATASET}/test.extxyz \
    --out-path /home/workspace/MLFF/HfO_v1.0/${DATASET} \
    --r-max 6.0 \
    --max-neighbors 50 


