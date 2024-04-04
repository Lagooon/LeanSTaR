set -e
set -x

export DATA_DIR=/workspace/zhiqings/output4/math
export MODEL_REPO=deepseek-ai/deepseek-math-7b-instruct

python scripts/download.py \
    --repo_id $MODEL_REPO \
    --local_dir $DATA_DIR/checkpoints

ls data_utils
ls grading
ls models
ls training_utils

python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO \
    --target_precision bf16
