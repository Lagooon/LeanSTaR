set -e
set -x

export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-plus-7b
export OMP_NUM_THREADS=8
export NCCL_IGNORE_DISABLED_P2P=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 2048 \
    --target_max_len 2048 \
    --total_max_len 2048 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 2 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --dataset "data/leandojo_benchmark_4/processed/proofstep-train.json" \
    --dataset_format "mapped" \
    --add_eos_to_target \
    --save_strategy epoch \
    --save_steps 60\
    --save_total_limit 3 \
    --save_dir $DATA_DIR/checkpoints/internlm2-7b_sft_epoch-3_lr-3e-5-plus 
    #--resume_from_checkpoint

# --dataset "alpaca" \
