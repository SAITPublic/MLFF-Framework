#!/bin/bash


MODEL_ARRAY=(
    "BPNN"
    "SchNet"
    "DimeNet++"
    "GemNet-T"
    "GemNet-dT"
    "NequIP"
    "Allegro"
    "MACE"
    "SCN"
)

DATA_ARRAY=(
    "SiN"
    "HfO"
)

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

MODEL="NequIP"
DATA="HfO"

CONFIG=configs/train/${DATA}/${MODEL}.yml
RUNDIR=train_results/${DATA}/${MODEL}
RUNID=train

GPU="0,1,2,3,4,5,6,7"
NUMGPUS=8
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$NUMGPUS main.py \
    --distributed \
    --num-gpus $NUMGPUS \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --identifier $RUNID \
    --print-every 100 \
    --save-ckpt-every-epoch 20 \
