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
    finetune_rm.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 16 \
    --learning_rate 2e-5 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 2 \
    --dataset "/nobackup/users/yikangs/zhiqings/math/processed_shepherd_v1_level1-3.json" \
    --dataset_format "prm-v4" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model True \
    --train_on_every_token \
    --save_dir $DATA_DIR/checkpoints/llemma-7b-rm_toet_shepherd-1-2-3-v1_epoch-2_lr-2e-5_seq-768 \
    --resume_from_checkpoint
