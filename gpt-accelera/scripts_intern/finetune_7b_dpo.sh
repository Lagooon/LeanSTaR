set -e
set -x

export DATA_DIR=/nobackup/users/zhiqings/haohanl/Lean
export MODEL_REPO=internlm/internlm2-math-base-7b
export OMP_NUM_THREADS=8
export GITHUB_ACCESS_TOKEN="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"
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

# per_device_train_batch_size = accumulate_steps * micro_batch_size

#    --param_dtype fp32 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345\
    finetune_dpo.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --sft_checkpoint_path $DATA_DIR/checkpoints/internlm2-7b_star_epoch-2_lr-3e-5 \
    --source_max_len 2048 \
    --target_max_len 2048 \
    --total_max_len 2048 \
    --per_device_train_batch_size 8 \
    --micro_train_batch_size 1 \
    --learning_rate 3e-6 \
    --lr_eta_min 3e-8 \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --optim_dtype fp32 \
    --dataset "data/leandojo_benchmark_4/processed/cot-generated-1.json" \
    --save_strategy epoch \
    --save_steps 60\
    --save_total_limit 2 \
    --dpo_beta 0.1 \
    --save_dir $DATA_DIR/checkpoints/internlm2-7b_dpo-_iter-1_lr-3e-6_beta-0.1_seq-1024_1 \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 1 \
    --print_training_examples False \
    --save_only_model True \
    --adam_beta2 0.95 \
    --adam_eps 1e-5 \
    --add_eos_to_target \
    #--resume_from_checkpoint
