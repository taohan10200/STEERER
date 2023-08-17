#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# -w SH-IDC1-10-140-1-163 \
set -x

#GPUS_PER_N
#GPUS=${GPUODE=${GPUS_PER_NODE:-4}

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=$4
# WORK_DIR=$5

GPUS=${GPUS}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
#SRUN_ARGS=${SRUN_ARGS:-"--quotatype=reserved"}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

source /mnt/petrelfs/hantao/anaconda3/bin/activate vit
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train_cc.py --cfg ${CONFIG} --launcher="slurm" ${PY_ARGS}