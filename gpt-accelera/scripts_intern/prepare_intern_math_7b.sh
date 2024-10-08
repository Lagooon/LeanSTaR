set -e
set -x

export HF_HOME=/data/user_data/shengyuf/Lean
export DATA_DIR=/data/user_data/shengyuf/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b

python scripts_intern/download.py \
    --repo_id $MODEL_REPO \
    --local_dir $DATA_DIR/checkpoints

python scripts_intern/convert_hf_checkpoint_intern.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO 
