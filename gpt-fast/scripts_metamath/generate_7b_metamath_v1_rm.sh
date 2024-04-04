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
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b-rm_toet_shepherd-1-2-3-v1_epoch-1_lr-2e-5_seq-768" \
    --batch_size 128 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_1-to-3_s256.jsonl" \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_7b-metamath_1-to-3_s256_scored-toet_epoch-1.jsonl" \
    --tensor_parallel_size 1 \
    --process_reward_with_answer \
    --resume_generation
