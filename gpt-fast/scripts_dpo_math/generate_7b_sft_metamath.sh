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
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_128_1_2_3_tp1_lr_min_5e-10_320acc" \
    --batch_size 32 \
    --max_new_tokens 1024 \
    --num_samples 8 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/train_1to5_1-2-3_prm_ppo.json" \
    --temperature 1.0 \
    --top_k 50 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to5_metamath_7b_320acc_r1_s8.jsonl" \
    --tensor_parallel_size 1 \
    --resume_generation

# on 1 x H100 tp=1
# 7b, bs = 32 (compile), 1700+ tokens/s
# 7b, bs = 64 (default_compile), 2200+ tokens/s

# on 1 x H100 tp=1, int8
# 7b, bs = 32  (compile), 1100+ tokens/s
# 7b, bs = 64 (default_compile), 1400+ tokens/s
