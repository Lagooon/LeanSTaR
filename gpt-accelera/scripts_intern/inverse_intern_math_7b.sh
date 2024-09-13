set -e
set -x

export DATA_DIR=/data/user_data/shengyuf/Lean
export MODEL_REPO=internlm/internlm2-math-plus-7b

#python scripts/download.py \
#    --repo_id $MODEL_REPO \
#    --local_dir $DATA_DIR/checkpoints

python scripts_intern/convert_hf_checkpoint_inverse.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO
