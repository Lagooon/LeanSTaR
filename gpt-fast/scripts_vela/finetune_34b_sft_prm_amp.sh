set -e
set -x

export DATA_DIR=/workspace/zhiqings/output4/math
export MODEL_REPO=EleutherAI/llemma_34b
export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

ls data_utils
ls grading
ls models
ls training_utils

# per_device_train_batch_size = accumulate_steps * micro_batch_size

torchrun --standalone --nproc_per_node=8 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 2 \
    --learning_rate 1e-5 \
    --lr_eta_min 1e-7 \
    --num_train_epochs 3 \
    --dataset "$DATA_DIR/train_1_2_3_prm_v4.json" \
    --dataset_format "prm-v2" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-34b_prm-1-2-3_epoch-3_lr-1e-5_seq-768-amp \
    --report_to "wandb" \
    --wandb_project "scalable-math-sft" \
    --wandb_entity "zhiqings" \
    --wandb_name "llemma-34b_prm-1-2-3_epoch-3_lr-1e-5_seq-768-amp" \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 4 \
    --adam_beta2 0.95 \
    --adam_eps 1e-5 \
    --weight_decay 0.1 \
    --save_only_model True
