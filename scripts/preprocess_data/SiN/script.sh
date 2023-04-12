#!/bin/bash

#DATADIR=/DB/SiN_v2.0

#python preprocess.py \
#    --train-data ${DATADIR}/Trainset_1.xyz \
#    --valid-data ${DATADIR}/Validset_1.xyz \
#    --test-data ${DATADIR}/Testset_1.xyz \
#    --out-path /home/workspace/MLFF/SiN_v2.0/split_1 \
#    --r-max 6.0 \
#    --max-neighbors 50 


#python preprocess.py \
#    --test-data ${DATADIR}/OOS.xyz \
#    --out-path /home/workspace/MLFF/SiN_v2.0/OOS \
#    --r-max 6.0 \
#    --max-neighbors 50 

DATADIR=/DB/SiN_v1.0
python preprocess.py \
    --train-data ${DATADIR}/xyz/Trainset.xyz \
    --valid-data ${DATADIR}/xyz/Validset.xyz \
    --test-data ${DATADIR}/xyz/Testset.xyz \
    --out-path /home/workspace/MLFF/SiN_v1.0 \
    #--r-max 5.0 

