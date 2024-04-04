set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=4 \
    batched_rm_score.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b-rm_sft-init_prm-1-2-3-v4_epoch-2_lr-1e-5_seq-768" \
    --batch_size 256 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s2048.jsonl" \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-v4_epoch-3_lr-2e-5_seq-768_s2048_scored-v1.jsonl" \
    --resume_generation

# on 4 x H100 tp=4
# 7b, bs = 256 (compile), 65000+ tokens/s

# on 1 x H100 tp=1
# 7b, bs = 256 (compile), 30000+ tokens/s
