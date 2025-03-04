#!/bin/bash

pip install wandb

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# 512 * 1k * 60k = 30b tokens.
TRAIN_ITERS=${2:-"60000"}
MICRO_BATCH_SIZE=${3:-"16"}
NUM_EXPERTS=${4:-"8"}
GRANILARITY=${5:-"1"}
PROJECT_NAME=train_llama_978m_remoe

CHECKPOINT_PATH=/workspace/weights/$PROJECT_NAME
mkdir -p $CHECKPOINT_PATH

PILE_DATASET="\
1.0 \
/workspace/dataset/pile_gpt2/00_text_document \
1.0 \
/workspace/dataset/pile_gpt2/01_text_document \
1.0 \
/workspace/dataset/pile_gpt2/02_text_document \
1.0 \
/workspace/dataset/pile_gpt2/03_text_document \
1.0 \
/workspace/dataset/pile_gpt2/04_text_document \
1.0 \
/workspace/dataset/pile_gpt2/05_text_document \
1.0 \
/workspace/dataset/pile_gpt2/06_text_document \
1.0 \
/workspace/dataset/pile_gpt2/07_text_document"

DISTRIBUTED_ARGS=(
    --nnodes $SLURM_NNODES
    --node_rank $SLURM_NODEID
    --nproc_per_node $SLURM_GPUS_ON_NODE
    --rdzv-id $SLURM_JOB_ID
    --rdzv-backend c10d
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 24
    --hidden-size 1536
    --ffn-hidden-size $((1536 * 4))
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
    --moe-relu-routing
    --moe-grouped-gemm
    # --moe-layer-recompute
    --moe-granularity $GRANILARITY
)

DATA_ARGS=(
    --vocab-file /workspace/dataset/gpt2-vocab.json \
    --merge-file /workspace/dataset/gpt2-merges.txt \
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
    --save-interval 2500
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

cd /workspace/ReMoE
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
