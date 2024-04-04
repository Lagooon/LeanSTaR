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

LEARNING_RATE=5e-5
BATCH_SIZE=16
GRAD_ACCUMULATION=1
NUM_EPOCHS=1
CKPT_STEPS=1000000

# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)
export MASTER_PORT=9901

ls models
ls data_utils

deepspeed \
    train_qlora_rm_ds.py \
    --deepspeed zero2.json \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path "EleutherAI/llemma_34b" \
    --learning_rate $LEARNING_RATE \
    --source_max_len 512 \
    --target_max_len 512 \
    --dataset_path "$FASTCHAT_DIR/data/math_rm/model_label/llemma-34b-lora-easy-sft-v4-ds-vllm-64.json" \
    --dataset_name "math_level1" \
    --eval_dataset_path "$FASTCHAT_DIR/data/math_rm/model_label/llemma-34b-lora-easy-sft-v4-ds-vllm-64.json" \
    --eval_dataset_name "math_level1" \
    --meta_prompt_pattern "../prompts/salmon_reward_model_prompt_math.txt" \
    --double_quant True \
    --quant_type "nf4" \
    --bits 8 \
    --lora_r 64 \
    --output_dir "$MODEL_DIR/llemma-34b-lora-easy-rm-v2-ds" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps $CKPT_STEPS \
    --save_total_limit 3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "wandb" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --add_eos_to_target False \
    --single_reward_model True \
    --label_names "choice"
