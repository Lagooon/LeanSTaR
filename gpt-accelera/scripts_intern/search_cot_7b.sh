set -e
set -x

export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b

export OMP_NUM_THREADS=4
export GITHUB_ACCESS_TOKEN="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"
# export ENABLE_INTRA_NODE_COMM=1
export TOKENIZERS_PARALLELISM=false

MAX_ITERS=5
NUM_SAMPLES=32
TEMPERATURES=0.7
TIMEOUT=20
NUM_SHARDS=4
DATASET="leandojo"
DATA="data/leandojo_benchmark_4/leandojo_benchmark_4/novel_premises/train.json"
export CONTAINER="native"
export TMP_DIR="/tmp/lean"

export CACHE_DIR="/tmp/.cache/dojo"

MODEL="internlm/internlm2-math-base-7b"
NAME="internLM2-7b"

OUTPUT_DIR="output/${NAME}_mathlib_test"
mkdir -p logs
#rm -rf /tmp/tmp*
#--finetune_checkpoint_path "$DATA_DIR/checkpoints/internlm2-7b_dpo-_epoch-1_lr-3e-6_beta-0.1_seq-1024_2" \
    
for SHARD in {0..3}
do
	CUDA_VISIBLE_DEVICES=$((SHARD)) python cot_search.py \
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
    --batch_size 16 \
    --top_k 200 \
    &> logs/${NAME}_shard${SHARD}.out &
done
