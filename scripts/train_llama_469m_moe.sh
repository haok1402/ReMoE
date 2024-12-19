#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=${1:-"1"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 512 * 1k * 60k = 30b tokens.
TRAIN_ITERS=${2:-"60000"}
MICRO_BATCH_SIZE=${3:-"16"}
NUM_EXPERTS=${4:-"8"}
GRANILARITY=${5:-"1"}
PROJECT_NAME=${6:-"test"}

CHECKPOINT_PATH="./logs/$PROJECT_NAME"
mkdir -p $CHECKPOINT_PATH

PILE_DATASET="\
1.0 \
../pile_gpt2/01_text_document \
1.0 \
../pile_gpt2/02_text_document \
1.0 \
../pile_gpt2/03_text_document \
1.0 \
../pile_gpt2/04_text_document \
1.0 \
../pile_gpt2/05_text_document \
1.0 \
../pile_gpt2/06_text_document \
1.0 \
../pile_gpt2/07_text_document \
1.0 \
../pile_gpt2/08_text_document \
1.0 \
../pile_gpt2/09_text_document \
1.0 \
../pile_gpt2/10_text_document \
1.0 \
../pile_gpt2/11_text_document \
1.0 \
../pile_gpt2/12_text_document \
1.0 \
../pile_gpt2/13_text_document \
1.0 \
../pile_gpt2/14_text_document \
1.0 \
../pile_gpt2/15_text_document \
1.0 \
../pile_gpt2/16_text_document \
1.0 \
../pile_gpt2/17_text_document \
1.0 \
../pile_gpt2/18_text_document \
1.0 \
../pile_gpt2/19_text_document \
1.0 \
../pile_gpt2/20_text_document \
1.0 \
../pile_gpt2/21_text_document \
1.0 \
../pile_gpt2/22_text_document \
1.0 \
../pile_gpt2/23_text_document \
1.0 \
../pile_gpt2/24_text_document \
1.0 \
../pile_gpt2/25_text_document \
1.0 \
../pile_gpt2/26_text_document \
1.0 \
../pile_gpt2/27_text_document \
1.0 \
../pile_gpt2/28_text_document \
1.0 \
../pile_gpt2/29_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size $((1024 * 4))
    --num-attention-heads 16
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --use-flash-attn
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --moe-router-topk 1
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-router-pre-softmax
    # --moe-relu-routing
    --moe-grouped-gemm
    # --moe-layer-recompute
    --moe-granularity $GRANILARITY
)

DATA_ARGS=(
    --vocab-file ../gpt2-vocab.json \
    --merge-file ../gpt2-merges.txt \
    --make-vocab-size-divisible-by 1024 \
    --data-path $PILE_DATASET
    --split 969,30,1
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size 512
    --lr 5e-4
    --train-iters $TRAIN_ITERS
    --lr-decay-style cosine
    --min-lr 5e-5
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 10
    --log-throughput 
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 100
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "ReMoE"
        --wandb-exp-name $PROJECT_NAME
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} |& tee -a $CHECKPOINT_PATH/train.log
