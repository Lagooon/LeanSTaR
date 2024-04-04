set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/dpo
export MODEL_REPO=huggyllama/llama-7b

python scripts/download.py \
    --repo_id $MODEL_REPO \
    --local_dir $DATA_DIR/checkpoints

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO \
    --target_precision bf16
