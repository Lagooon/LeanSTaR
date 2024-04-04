set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/workspace/zhiqings/output3/models"
export OMP_NUM_THREADS=8

MODEL_NAME="llemma-34b-lora-easy-sft-v4-ds"

# python gen_model_answer_vllm.py \

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    gen_model_answer_vllm.py \
    --bench-name "math_rm" \
    --model-path "$MODEL_DIR/$MODEL_NAME-merged" \
    --num-choices 64 \
    --temperature 0.7 \
    --model-id "$MODEL_NAME-vllm-64" \
    --num-gpus-per-model 1 \
    --model-prompt "dromedary" \
    --max-new-token 512 \
    --num-gpus-total 8 \
    --num-gpus-per-model 1
