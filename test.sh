# !/usr/bin/env sh

# set -x

CONFIG=$1
CHECKPOINT=$2
GPUS_ID=${3:-0} 

PORT=${4:-29000}   #the default gpu_id is 1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

source ~/anaconda3/bin/activate STEERER



GPU_NUM=0

for ((i=0; i<${#GPUS_ID}; i++)); do
    if [[ ${GPUS_ID:i:1} =~ [0-9] ]]; then
        ((GPU_NUM++))
    fi
done
export CUDA_VISIBLE_DEVICES=${GPUS:-"1"}
# torchrun --nproc_per_node=${GPU_NUM} --master_port ${PORT} tools/train_cc.py --cfg ${CONFIG} 

python -m torch.distributed.launch \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPU_NUM \
    --master_port=$PORT \
    tools/test_loc.py --cfg=$CONFIG --checkpoint=${CHECKPOINT} --launcher="pytorch"
