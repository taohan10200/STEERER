CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29000}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
source /mnt/petrelfs/hantao.dispatch/anaconda3/bin/activate nwp

echo "export CUDA_VISIBLE_DEVICES=$4"
export CUDA_VISIBLE_DEVICES=${4:-"1,2"}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

torchrun --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        tools/test_cc.py \
        --cfg=$CONFIG \
        --checkpoint=$CHECKPOINT \
        --launcher pytorch ${@:5}