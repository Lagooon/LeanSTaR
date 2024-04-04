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
    finetune_dpo.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --total_max_len 1024 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 16 \
    --learning_rate 2e-6 \
    --lr_eta_min 2e-8 \
    --warmup_ratio 0.2 \
    --num_train_epochs 2 \
    --dataset "/nobackup/users/yikangs/zhiqings/dpo/alpaca_rm_10k.json" \
    --dataset_format "alpaca" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/alpaca-7b-dpo_epoch-2_lr-2e-6_seq-1024 \
    --resume_from_checkpoint \
    --report_to "wandb" \
    --wandb_project "scalable-alpacafarm" \
    --wandb_entity "zhiqings" \
    --wandb_name "alpaca-7b-dpo_epoch-2_lr-2e-6_seq-1024" \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 1 \
    --print_training_examples True \
    --save_only_model True \
    --adam_beta2 0.95 \
    --adam_eps 1e-5 \
    --add_eos_to_target True
