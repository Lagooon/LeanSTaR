set -e
set -x

# cd Dromedary/training
cd SALMON/training

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/workspace/zhiqings/output3/models"
export DATA_DIR="/workspace/zhiqings/output3/data"
export FASTCHAT_DIR="/workspace/zhiqings/ScalableMath/fastchat_eval"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo $PYTHONPATH

export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)
export MASTER_PORT=9901

ls models
ls data_utils

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    batched_rm_scoring.py \
    --input_question_path "$FASTCHAT_DIR/data/math/question.jsonl" \
    --input_sampling_path "$FASTCHAT_DIR/data/math/model_answer/llemma-34b-lora-easy-sft-v4-ds-vllm-64.jsonl" \
    --output_scoring_path "$FASTCHAT_DIR/data/math/model_answer/llemma-34b-lora-easy-sft-v4-ds-vllm-64-scored-pretrain-v4.jsonl" \
    --model_path "$MODEL_DIR/llemma-34b-lora-easy-pretrain-rm-v4-ds" \
    --meta_prompt_path "../prompts/salmon_reward_model_prompt_math.txt" \
    --per_device_batch_size 4
