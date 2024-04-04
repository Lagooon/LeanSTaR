set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 256 \
    --per_device_train_batch_size 64 \
    --micro_train_batch_size 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --dataset "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_prm.json" \
    --dataset_format "prm" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-7b_prm-1-2-3_epoch-3_lr-1e-4 \
    --resume_from_checkpoint

# --dataset "alpaca" \
