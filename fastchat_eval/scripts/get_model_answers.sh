set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/workspace/zhiqings/output3/models"

MODEL_NAME="llemma-34b-lora-easy-pretrain-sft-v3-ds"

python gen_model_answer.py \
    --bench-name "math" \
    --model-path "EleutherAI/llemma_34b" \
    --qlora-path "$MODEL_DIR/$MODEL_NAME/lora_default" \
    --qlora-bits 16 \
    --model-id "$MODEL_NAME" \
    --num-gpus-per-model 1 \
    --model-prompt "dromedary" \
    --max-new-token 512 \
    --num-gpus-total 8
