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
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b-rm_toet_shepherd-1-2-3-v1_epoch-2_lr-2e-5_seq-768" \
    --batch_size 128 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_metamath_7b_320acc_t1.2_r1_s8.jsonl" \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/train_1to3_metamath_7b_320acc_t1.2_r1_s8_scored-toet.jsonl" \
    --tensor_parallel_size 1 \
    --resume_generation
