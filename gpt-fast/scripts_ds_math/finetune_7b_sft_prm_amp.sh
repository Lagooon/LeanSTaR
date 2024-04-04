set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=deepseek-ai/deepseek-math-7b-base
export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 16 \
    --learning_rate 1e-5 \
    --lr_eta_min 1e-7 \
    --num_train_epochs 3 \
    --dataset "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_prm_v4.json" \
    --dataset_format "prm-v2" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_only_model True \
    --save_total_limit 1 \
    --tensor_parallel_size 1 \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --report_to "wandb" \
    --save_dir $DATA_DIR/checkpoints/deepseek-math-7b_prm-1-2-3-v4_epoch-3_lr-2e-5_seq-768-amp \
    --wandb_project "scalable-math-sft" \
    --wandb_entity "zhiqings" \
    --wandb_name "deepseek-math-7b_prm-1-2-3-v4_epoch-3_lr-2e-5_seq-768-amp" \
    --resume_from_checkpoint
