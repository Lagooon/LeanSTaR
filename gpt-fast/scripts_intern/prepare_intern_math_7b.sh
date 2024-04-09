set -e
set -x

export HF_HOME=/nobackup/users/zhiqings/haohanl/Lean
export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b

python scripts/download.py \
    --repo_id $MODEL_REPO \
    --local_dir $DATA_DIR/checkpoints

python scripts_intern/convert_hf_checkpoint_intern.py \
    --checkpoint_dir $DATA_DIR/checkpoints/$MODEL_REPO 
