set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=8 \
    batched_rm_score.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b-rm_sft-init-toet_prm-1-2-3-v4_epoch-1_lr-2e-5_seq-768" \
    --batch_size 128 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_prm_7b-v4_epoch-3_lr-2e-5_seq-768_r1_s8.jsonl" \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_prm_7b-v4_epoch-3_lr-2e-5_seq-768_r1_s8_scored-toet.jsonl" \
    --tensor_parallel_size 1 \
    --resume_generation
