export MODEL_PATH='/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/EleutherAI/llemma_34b'
export SAVE_PATH="/lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma-34b_metamath_v3_5e-6_1_2_3_4_5"
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
module load cuda/12.1
export WANDB_DISABLED=true
wandb disabled

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_llemma_34b.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/train_1_2_3_4_5_metamath_v3_mapped.json" \
    --data_length 100000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 2 \
    --model_max_length 768 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 0 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --deepspeed ./zero2_config_30b.json \
    --tf32 True
