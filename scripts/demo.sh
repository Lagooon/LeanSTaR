set -e
set -x

cd inference

python -u demo.py \
    --model-name "EleutherAI/llemma_34b" \
    --adapters-name "/workspace/zhiqings/output3/models/llemma-34b-lora-easy-pretrain-sft-v3-ds/lora_default"
