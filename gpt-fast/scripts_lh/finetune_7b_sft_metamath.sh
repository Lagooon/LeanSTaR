set -e
set -x

export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
export ENABLE_INTRA_NODE_COMM=1
module load cuda/11.8
export HF_HOME=/tmp
# export NCCL_IB_TIMEOUT=22
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
    --per_device_train_batch_size 128 \
    --micro_train_batch_size 8 \
    --learning_rate 1e-5 \
    --lr_eta_min 1e-7 \
    --num_train_epochs 3 \
    --dataset "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/metamath_math_level1-3_20193_mapped.json" \
    --dataset_format "mapped" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma_7b_metamath_3_epochs_1e-5
 
# /lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma_7b_6e-6_metamath_mapped