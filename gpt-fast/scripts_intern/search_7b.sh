set -e
set -x

export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b

export OMP_NUM_THREADS=4
export GITHUB_ACCESS_TOKEN="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"
# export ENABLE_INTRA_NODE_COMM=1

MAX_ITERS=100
NUM_SAMPLES=32
TEMPERATURES=0.0
TIMEOUT=600
NUM_SHARDS=4
DATASET="minif2f-test"
DATA="data/minif2f.jsonl"
export CONTAINER="native"
export TMP_DIR="/nobackup/users/zhiqings/haohanl/tmp"

MODEL="internlm/internlm2-math-base-7b"
NAME="internLM2-7b"

OUTPUT_DIR="output/${NAME}_minif2f_test"
mkdir -p logs
#rm -rf /tmp/tmp*
#--finetune_checkpoint_path "$DATA_DIR/checkpoints/internlm2-7b_dpo-_epoch-1_lr-3e-6_beta-0.1_seq-1024_2" \
    
for SHARD in 4
do
    CUDA_VISIBLE_DEVICES=${SHARD} python batched_search.py \
    --dataset-name ${DATASET} \
    --temperature ${TEMPERATURES} \
    --timeout ${TIMEOUT} \
    --num-shards ${NUM_SHARDS} \
    --shard ${SHARD} \
    --shard-base 4 \
    --max-iters ${MAX_ITERS} \
    --dataset-path ${DATA} \
    --num-samples ${NUM_SAMPLES} \
    --early-stop \
    --output-dir ${OUTPUT_DIR} \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --batch_size 16 \
    --top_k 200 \
    &> logs/${NAME}_shard${SHARD}.out &
done
