set -e
set -x

export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b
export OMP_NUM_THREADS=8
export NCCL_IGNORE_DISABLED_P2P=1
ITER=1
SPLIT=4
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

#torchrun --standalone --nproc_per_node=8 \
for SHARD in 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=${SHARD} python generate_snip.py \
        --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/finetune \
        --source_max_len 896 \
        --target_max_len 128 \
        --seed 43\
        --per_device_train_batch_size 8 \
        --iter ${ITER} \
        --split ${SPLIT} \
        --frac ${SHARD} \
        --dataset "data/leandojo_benchmark_4/processed/proofstep-train.json" \
        --dataset_format "mapped" \
    &> logs/${NAME}_spingen${SHARD}.out &
done

# --dataset "alpaca" \
