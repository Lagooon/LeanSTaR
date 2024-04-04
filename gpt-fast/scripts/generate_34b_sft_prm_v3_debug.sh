set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_34b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=8 \
    batched_generate.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-34b_prm-1-2-3_epoch-3_lr-1e-5_seq-768-yk" \
    --batch_size 64 \
    --max_new_tokens 1024 \
    --num_samples 1 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
    --temperature 0.0 \
    --top_k 1 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-v4_epoch-3_lr-2e-5_seq-768_g.jsonl" \
    --resume_generation

# on 2 x H100 tp=2
# 34b, bs = 64 (compile), 900 tokens/s
# 34b, bs = 64 (default_compile), 900 tokens/s

# on 1 x H100 tp=1, int8
# 34b, bs = 32 (default_compile), 280 tokens/s
# 34b, bs = 64 (default_compile), 380 tokens/s
