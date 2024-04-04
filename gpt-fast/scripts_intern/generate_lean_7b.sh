set -e
set -x

export DATA_DIR=/localdata_ssd/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b

export OMP_NUM_THREADS=4
export GITHUB_ACCESS_TOKEN="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"
# export ENABLE_INTRA_NODE_COMM=1

MAX_ITERS=100
NUM_SAMPLES=64
TEMPERATURES=0.5
TIMEOUT=300
NUM_SHARDS=8
DATASET="minif2f-test"
DATA="data/minif2f.jsonl"

MODEL="internlm/internlm2-math-base-7b"
NAME="internLM2-7b"

OUTPUT_DIR="output/${NAME}_minif2f_test"
mkdir -p logs
rm -rf /tmp/tmp*
for SHARD in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=${SHARD} python batched_generate_lean.py \
    --dataset-name ${DATASET} \
    --temperature ${TEMPERATURES} \
    --timeout ${TIMEOUT} \
    --num-shards ${NUM_SHARDS} \
    --shard ${SHARD} \
    --shard-base 0 \
    --max-iters ${MAX_ITERS} \
    --dataset-path ${DATA} \
    --num-samples ${NUM_SAMPLES} \
    --early-stop \
    --output-dir ${OUTPUT_DIR} \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/internlm2-7b_sft_epoch-4_lr-3e-5_1" \
    --batch_size 8 \
    --top_k 500 \
  &> logs/${NAME}_shard${SHARD}.out &
done
