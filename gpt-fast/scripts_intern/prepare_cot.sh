SIZE=1000
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

#torchrun --standalone --nproc_per_node=8 \
for START in 0 1000 2000 3000 4000 5000 6000 7000 8000 9000
do
    python scripts_intern/gen.py \
        --start ${START} \
        --size ${SIZE} \
    &> logs/gptgen${START}.md &
done

# --dataset "alpaca" \
