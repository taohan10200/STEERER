# !/usr/bin/env sh

# set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

source /mnt/petrelfs/hantao.dispatch/anaconda3/bin/activate STEERER

export CUDA_VISIBLE_DEVICES=${GPUS:-"1,2"}

python  tools/test_cc.py \
        --cfg=$CONFIG \
        --checkpoint=$CHECKPOINT \
