set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/dpo
export MODEL_REPO=huggyllama/llama-7b
export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

torchrun --standalone --nproc_per_node=4 \
    finetune_pm_pairwise.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --sft_checkpoint_path $DATA_DIR/checkpoints/alpaca-7b-10k_epoch-3_lr-1e-4_seq-768 \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 32 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --dataset "/nobackup/users/yikangs/zhiqings/dpo/hh_rlhf_pm_10k.json" \
    --dataset_format "alpaca" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/hh-rlhf-7b-pm_epoch-1_lr-5e-6_seq-2048 \
    --resume_from_checkpoint \
    --report_to "wandb" \
    --wandb_project "scalable-alpacafarm" \
    --wandb_entity "zhiqings" \
    --wandb_name "hh-rlhf-7b-pm_epoch-1_lr-5e-6_seq-2048" \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 1 \
    --print_training_examples True \
    --save_only_model True \
    --reward_head_init_scheme semantic
