set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=4 \
    batched_generate.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b-ppo_prm-1to5_epoch-50_lr-2e-6_seq-768-vj" \
    --batch_size 32 \
    --max_new_tokens 1024 \
    --num_samples 1 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
    --temperature 0.7 \
    --top_k 1 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_ppo_prm_v3_7b-1to3-vj_epoch-50_lr-2e-5_seq-768_g.jsonl" \
    --tensor_parallel_size 1 \
    --finetune_checkpoint_prefix "policy_" \
    --resume_generation

# on 1 x H100 tp=1
# 7b, bs = 32 (compile), 1700+ tokens/s
# 7b, bs = 64 (default_compile), 2200+ tokens/s

# on 1 x H100 tp=1, int8
# 7b, bs = 32  (compile), 1100+ tokens/s
# 7b, bs = 64 (default_compile), 1400+ tokens/s
