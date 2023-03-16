#!/bin/bash

DATADIR=/DB/rmd17/xyz_data
OUTPUTDIR=/home/workspace/MLFF/rmd17
SPLITDIR=/nas/NeurIPS23_DnB/split/rMD17

MOLECULARS=("aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde", "naphthalene", "paracetamol", "salicylic", "toluene", "uracil")


for mol in "${MOLECULARS[@]}"; do

INPUT_XYZ=${DATADIR}/rmd17_${mol}.xyz
TRAIN_IDX=${SPLITDIR}/rmax_1/rmd17_${mol}_train.txt
VAL_IDX=${SPLITDIR}/rmax_1/rmd17_${mol}_val.txt
TEST_IDX=${SPLITDIR}/rmax_1/rmd17_${mol}_test.txt

python convert_extxyz_to_graph_lmdb.py --data-path $INPUT_XYZ --r-max 6.0 --out-path $OUTPUTDIR/rmax6_1 --train-index $TRAIN_IDX --val-index $VAL_IDX --test-index $TEST_IDX

done


for mol in "${MOLECULARS[@]}"; do

INPUT_XYZ=${DATADIR}/rmd17_${mol}.xyz
TRAIN_IDX=${SPLITDIR}/rmax_1/rmd17_${mol}_train.txt
VAL_IDX=${SPLITDIR}/rmax_1/rmd17_${mol}_val.txt
TEST_IDX=${SPLITDIR}/rmax_1/rmd17_${mol}_test.txt

python convert_extxyz_to_graph_lmdb.py --data-path $INPUT_XYZ --r-max 6.0 --out-path $OUTPUTDIR/rmax6_1 --train-index $TRAIN_IDX --val-index $VAL_IDX --test-index $TEST_IDX

done

