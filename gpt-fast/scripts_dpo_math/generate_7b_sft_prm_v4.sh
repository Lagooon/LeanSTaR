set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=8 \
    batched_generate.py \
    --default_compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_prm-1-2-3-v4_epoch-3_lr-2e-5_seq-768" \
    --batch_size 32 \
    --max_new_tokens 1024 \
    --num_samples 8 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/train_1to3_1-2-3_prm_ppo.json" \
    --temperature 1.0 \
    --top_k 50 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_prm_7b-v4_epoch-3_lr-2e-5_seq-768_r1_s8.jsonl" \
    --tensor_parallel_size 1 \
    --resume_generation

# on 1 x H100 tp=1
# 7b, bs = 32 (compile), 1700+ tokens/s
# 7b, bs = 64 (default_compile), 2200+ tokens/s

# on 1 x H100 tp=1, int8
# 7b, bs = 32  (compile), 1100+ tokens/s
# 7b, bs = 64 (default_compile), 1400+ tokens/s
