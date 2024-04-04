set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=EleutherAI/llemma_34b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=4 \
    batched_generate.py \
    --default_compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-34b_metamath_v6_pruned_lr-8e-6_1to5_epoch1_amp" \
    --batch_size 32 \
    --max_new_tokens 1024 \
    --num_samples 256 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3.json" \
    --temperature 0.7 \
    --top_k 20 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_34b-metamath-8e-6_v6_amp_1-to-5_s256.jsonl" \
    --tensor_parallel_size 4 \
    --resume_generation
