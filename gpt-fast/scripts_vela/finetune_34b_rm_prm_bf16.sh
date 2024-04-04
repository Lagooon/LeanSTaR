set -e
set -x

export DATA_DIR=/workspace/zhiqings/output4/math
export MODEL_REPO=EleutherAI/llemma_34b
export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1
export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300

ls data_utils
ls grading
ls models
ls training_utils

# per_device_train_batch_size = accumulate_steps * micro_batch_size

torchrun --standalone --nproc_per_node=8 \
    finetune_rm.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --sft_checkpoint_path $DATA_DIR/checkpoints/llemma-34b_prm-1-2-3_epoch-3_lr-1e-5_seq-768-bf16 \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 4 \
    --learning_rate 1e-5 \
    --lr_eta_min 1e-7 \
    --num_train_epochs 2 \
    --dataset "/workspace/zhiqings/output4/math/train_1_2_3_prm_v4_rm.json" \
    --dataset_format "prm-v3" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --train_on_every_token \
    --save_dir $DATA_DIR/checkpoints/llemma-34b-rm_sft-init-toet_prm-1-2-3-v4_epoch-2_lr-1e-5_seq-768-bf16 \
    --report_to "wandb" \
    --wandb_project "scalable-math-sft" \
    --wandb_entity "zhiqings" \
    --wandb_name "llemma-34b_rm-sft-init-toet-1-2-3_epoch-2_lr-1e-5_seq-768-bf16" \
    --optimizer_cpu_offload True \
    --resume_from_checkpoint
