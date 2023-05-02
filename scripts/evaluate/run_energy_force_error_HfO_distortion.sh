#!/bin/bash

GPU=$1
CKPT=$2
#DATA=/DB/HfO_v1.0/symmetric_distortion
DATA=$3

./run_energy_force_error.sh $GPU $CKPT ${DATA}/test_0.01.extxyz
./run_energy_force_error.sh $GPU $CKPT ${DATA}/test_0.03.extxyz
./run_energy_force_error.sh $GPU $CKPT ${DATA}/test_0.09.extxyz
./run_energy_force_error.sh $GPU $CKPT ${DATA}/test_0.27.extxyz
./run_energy_force_error.sh $GPU $CKPT ${DATA}/test_0.81.extxyz
