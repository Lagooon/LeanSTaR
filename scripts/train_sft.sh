set -e
set -x

cd Dromedary/training

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/workspace/zhiqings/output3/models"
export DATA_DIR="/workspace/zhiqings/output3/data"
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

LEARNING_RATE=1e-4
BATCH_SIZE=8
GRAD_ACCUMULATION=2
NUM_EPOCHS=1
CKPT_STEPS=500

# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)
export MASTER_PORT=9901

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_qlora.py \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path "EleutherAI/llemma_34b" \
    --learning_rate $LEARNING_RATE \
    --source_max_len 512 \
    --target_max_len 512 \
    --dataset "$DATA_DIR/train_sft.json" \
    --dataset_format "dromedary" \
    --meta_prompt_pattern "../prompts/inference_prompts/dromedary_*prompt_distill.txt" \
    --double_quant True \
    --quant_type "nf4" \
    --bits 8 \
    --lora_r 64 \
    --output_dir "$MODEL_DIR/llemma-34b-lora-easy-sft" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $CKPT_STEPS \
    --save_total_limit 3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --fp16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --add_eos_to_target False
