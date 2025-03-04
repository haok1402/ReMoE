#!/bin/bash
# Run the processing inside the container.

cd /workspace/ReMoE
DATASET=/workspace/dataset

for i in $(seq -w 00 07); do
    python tools/preprocess_data.py \
        --input $DATASET/pile/${i}.jsonl \
        --output-prefix $DATASET/pile_gpt_test/${i} \
        --vocab-file $DATASET/gpt2-vocab.json \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file $DATASET/gpt2-merges.txt \
        --append-eod \
        --workers 32
done
