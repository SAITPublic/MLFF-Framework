#!/bin/bash

BENCHMARK_HOME=$(realpath ../../../)
DATADIR=${BENCHMARK_HOME}/datasets/SiN
OUTDIR=${BENCHMARK_HOME}/datasets/SiN

cd ..

# If you want to prepare .lmdb which saves just atom cloud (containing just coordinates), set cloud.
# Or if you want to have graph (containing coordinates as well as edges), set graph
outdata_type=cloud

if [ $outdata_type == "cloud" ]; then

# SiN Train/Valid/Test sets
python preprocess.py \
    --train-data ${DATADIR}/Trainset_1.xyz \
    --valid-data ${DATADIR}/Validset_1_shuffled.xyz \
    --valid-data-output-name valid_shuffled \
    --test-data ${DATADIR}/Testset_1_shuffled.xyz \
    --test-data-output-name test_shuffled \
    --out-path ${OUTDIR}/split_1 \

# SiN OOD
python preprocess.py \
    --data ${DATADIR}/OOS.xyz \
    --data-output-name ood \
    --out-path ${OUTDIR}/ood \

elif [ $outdata_type == "graph" ]; then

# SiN Train/Valid/Test sets
python preprocess.py \
    --train-data ${DATADIR}/Trainset_1.xyz \
    --valid-data ${DATADIR}/Validset_1_shuffled.xyz \
    --valid-data-output-name valid_shuffled \
    --test-data ${DATADIR}/Testset_1_shuffled.xyz \
    --test-data-output-name test_shuffled \
    --out-path ${OUTDIR}/split_1 \
    --r-max 6.0 \
    --max-neighbors 50 \

# SiN OOD
python preprocess.py \
    --data ${DATADIR}/OOS.xyz \
    --data-output-name ood \
    --out-path ${OUTDIR}/ood \
    --r-max 6.0 \
    --max-neighbors 50 \


fi
