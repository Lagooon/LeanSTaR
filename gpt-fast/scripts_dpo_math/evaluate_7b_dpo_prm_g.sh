set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=4 \
    batched_generate.py \
    --default_compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_dpo-prm_1-to-5_epoch-5_lr-3e-6_beta-0.1_seq-2048-r2" \
    --batch_size 32 \
    --max_new_tokens 1024 \
    --num_samples 1 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
    --temperature 0.0 \
    --top_k 1 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-dpo-prm_epoch-5_v1-r2_g.jsonl" \
    --tensor_parallel_size 1 \
    --resume_generation

# on 1 x H100 tp=1
# 7b, bs = 32 (compile), 1700+ tokens/s
# 7b, bs = 64 (default_compile), 2200+ tokens/s

# on 1 x H100 tp=1, int8
# 7b, bs = 32  (compile), 1100+ tokens/s
# 7b, bs = 64 (default_compile), 1400+ tokens/s
