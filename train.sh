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
GPUS_ID=${2:-0}    #the default gpu_id is 0 
PORT=${3:-29000}   #the default gpu_id is 1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

GPU_NUM=0

for ((i=0; i<${#GPUS_ID}; i++)); do
    if [[ ${GPUS_ID:i:1} =~ [0-9] ]]; then
        ((GPU_NUM++))
    fi
done

source /mnt/petrelfs/hantao.dispatch/anaconda3/bin/activate STEERER

echo "export CUDA_VISIBLE_DEVICES=$GPUS_ID"
export CUDA_VISIBLE_DEVICES=${GPUS_ID:-"0"}


torchrun --nproc_per_node=${GPU_NUM} --master_port ${PORT} tools/train_cc.py --cfg ${CONFIG} 

# --launcher="pytorch"
# python -m torch.distributed.launch \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPU_NUM \
#     --master_port=$PORT \
#     tools/train_cc.py --cfg=$CONFIG --launcher="pytorch"