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
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-34b_prm-1-2-3_epoch-3_lr-1e-5_seq-768" \
    --batch_size 64 \
    --max_new_tokens 1024 \
    --num_samples 64 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1_2_3_prm_v3.json" \
    --temperature 0.5 \
    --top_k 20 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_v3_34b_epoch-3_lr-2e-5_seq-768_s64.jsonl" \
    --resume_generation
