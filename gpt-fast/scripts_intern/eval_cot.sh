set -e
set -x

export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b
export OMP_NUM_THREADS=8
export NCCL_IGNORE_DISABLED_P2P=1
ITER=2
SPLIT=8
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

#torchrun --standalone --nproc_per_node=8 \
for SHARD in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=${SHARD} python eval_cot.py \
        --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/cots \
        --source_max_len 2048 \
        --target_max_len 2048 \
        --seed 43\
        --per_device_train_batch_size 4 \
        --iter ${ITER} \
        --split ${SPLIT} \
        --frac ${SHARD} \
        --dataset "data/leandojo_benchmark_4/processed/proofstep-train.json" \
        --dataset_format "mapped" \
    &> logs/COT_eval${SHARD}.out &
done

# --dataset "alpaca" \

