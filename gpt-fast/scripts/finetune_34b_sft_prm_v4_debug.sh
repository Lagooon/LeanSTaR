set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_34b
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
    --learning_rate 1e-5 \
    --lr_eta_min 1e-7 \
    --num_train_epochs 3 \
    --dataset "/nobackup/users/yikangs/zhiqings/math/train_1_2_3_metaprm_v4.json" \
    --dataset_format "prm-v2" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-34b_metaprm-1-2-3_epoch-3_lr-1e-5_seq-768-debug \
    --sequence_parallel \
    --optimizer_cpu_offload \
    --resume_from_checkpoint

# 0.4100720286369324
# 0.62525475025177
# 0.5716539025306702

# w/o activation_tensor_parallel (bs=1): 46954MiB
# w/o activation_tensor_parallel (bs=2): 52764MiB
# w/o activation_tensor_parallel (bs=4): 63964MiB
# w/  activation_tensor_parallel (bs=1): 41728MiB
# w/  activation_tensor_parallel (bs=2): 44472MiB
# w/  activation_tensor_parallel (bs=4): 50022MiB
