# !/usr/bin/env sh

# set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

source ~/anaconda3/bin/activate STEERER

# export CUDA_VISIBLE_DEVICES=${GPUS:-"1"}

python  tools/test_loc.py \
        --cfg=$CONFIG \
        --checkpoint=$CHECKPOINT \
        --launcher pytorch ${@:5}
