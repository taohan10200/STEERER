# !/usr/bin/env sh
# ${GPUS:-4}
set -x


# GPU_ID=$1
# CONFIG=$2
# GPU_NUM=$3

# echo "export CUDA_VISIBLE_DEVICES=${GPU_ID}"
# export CUDA_VISIBLE_DEVICES=${GPU_ID}


# if [ "${GPU_NUM}" -gt "1" ]; then
#   torchrun --nproc_per_node=${GPU_NUM} --master_port 29600 tools/train_cc.py --cfg ${CONFIG} --launcher="pytorch"
# else
#   python tools/train_cc.py --cfg ${CONFIG}
# fi


#!/usr/bin/env bash
# set -x

CONFIG=$1
GPUS=$2
WORK_DIR=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29000}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
source /mnt/petrelfs/hantao.dispatch/anaconda3/bin/activate nwp

echo "export CUDA_VISIBLE_DEVICES=$4"
export CUDA_VISIBLE_DEVICES=${4:-"1,2"}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_cc.py \
    --cfg=$CONFIG \
    --work-dir=$WORK_DIR \
    --launcher pytorch ${@:5}