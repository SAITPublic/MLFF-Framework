#!/bin/bash

DATADIR=/DB/SiN_v2.0
OUTDIR=/home/workspace/MLFF_DB/SiN_v2.0

cd ..

# split 1
python preprocess.py \
    --train-data ${DATADIR}/Trainset_1.xyz \
    --valid-data ${DATADIR}/Validset_1_shuffled.xyz \
    --valid-data-output-name valid_shuffled \
    --test-data ${DATADIR}/Testset_1_shuffled.xyz \
    --test-data-output-name test_shuffled \
    --out-path ${OUTDIR}/split_1 \
    # --r-max 6.0 \
    # --max-neighbors 50

# split 2
python preprocess.py \
    --train-data ${DATADIR}/Trainset_2.xyz \
    --valid-data ${DATADIR}/Validset_2_shuffled.xyz \
    --valid-data-output-name valid_shuffled \
    --test-data ${DATADIR}/Testset_2_shuffled.xyz \
    --test-data-output-name test_shuffled \
    --out-path ${OUTDIR}/split_2 \
    # --r-max 6.0 \
    # --max-neighbors 50

# OOD
python preprocess.py \
    --data ${DATADIR}/OOS.xyz \
    --data-output-name ood \
    --out-path ${OUTDIR}/ood \
    # --r-max 6.0 \
    # --max-neighbors 50
