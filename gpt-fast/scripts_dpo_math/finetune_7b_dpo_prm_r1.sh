set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

torchrun --standalone --nproc_per_node=4 \
    finetune_dpo.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --sft_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_prm-1-2-3-v4_epoch-3_lr-2e-5_seq-768" \
    --source_max_len 2048 \
    --target_max_len 2048 \
    --total_max_len 2048 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 1 \
    --learning_rate 3e-6 \
    --lr_eta_min 3e-8 \
    --warmup_ratio 0.03 \
    --num_train_epochs 2 \
    --dataset "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to5_prm_7b-v4_epoch-3_lr-2e-5_seq-768_r1_s8_dpo-toet_dup-3.json" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --dpo_beta 0.1 \
    --save_dir $DATA_DIR/checkpoints/llemma-7b_dpo-prm_1-to-5_epoch-3_lr-3e-6_beta-0.1_seq-2048 \
    --report_to "wandb" \
    --wandb_project "scalable-math-dpo" \
    --wandb_entity "zhiqings" \
    --wandb_name "llemma-7b_dpo-prm_1-to-5_epoch-3_lr-3e-6_beta-0.1_seq-2048" \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 1 \
    --print_training_examples True \
    --save_only_model True \
    --adam_beta2 0.95 \
    --adam_eps 1e-5 \
    --add_eos_to_marked_target True
