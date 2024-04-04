set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

# torchrun --standalone --nproc_per_node=4 \
#     batched_generate.py \
#     --compile \
#     --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
#     --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_prm-1to3-self-correct-v5-max-1_epoch-3_lr-2e-5_seq-768" \
#     --batch_size 32 \
#     --max_new_tokens 1024 \
#     --num_samples 2048 \
#     --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
#     --temperature 0.7 \
#     --top_k 20 \
#     --tensor_parallel_size 1 \
#     --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_self_correct_max-1_7b_s2048.jsonl" \
#     --resume_generation

torchrun --standalone --nproc_per_node=2 \
    batched_generate.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_prm-1to3-self-correct-v2-max-2_epoch-3_lr-2e-5_seq-768" \
    --batch_size 64 \
    --max_new_tokens 1024 \
    --num_samples 1 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
    --temperature 0.0 \
    --top_k 1 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1_2_3_prm_self_correct_v2_max-2_7b_g.jsonl" \
    --resume_generation
