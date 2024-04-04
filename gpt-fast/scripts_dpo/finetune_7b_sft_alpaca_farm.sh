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
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 16 \
    --learning_rate 2e-5 \
    --lr_eta_min 1e-6 \
    --num_train_epochs 3 \
    --dataset "/nobackup/users/yikangs/zhiqings/dpo/alpaca_sft_10k.json" \
    --dataset_format "alpaca" \
    --add_eos_to_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/alpaca-7b-10k_epoch-3_lr-1e-4_seq-768 \
    --resume_from_checkpoint \
    --report_to "wandb" \
    --wandb_project "scalable-alpacafarm" \
    --wandb_entity "zhiqings" \
    --wandb_name "alpaca-7b-10k_epoch-3_lr-1e-4_seq-768" \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 1 \
    --adam_eps 1e-5 \
    --resume_from_checkpoint
