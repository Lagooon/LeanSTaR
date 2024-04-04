set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/workspace/zhiqings/output3/models"

MODEL_NAME="llemma-34b-lora-easy-sft-v4-ds"

python merge_model.py \
    --adapters_name "$MODEL_DIR/$MODEL_NAME/lora_default" \
    --output_dir "$MODEL_DIR/$MODEL_NAME-merged"
