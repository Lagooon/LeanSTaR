set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/math
export MODEL_REPO=deepseek-ai/deepseek-math-7b-base
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=4 \
    batched_generate.py \
    --default_compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/deepseek-math-7b_mmiqc_v5_pruned_lr-8e-6_1to3_epoch1_amp/" \
    --batch_size 8 \
    --max_new_tokens 1024 \
    --num_samples 1 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/math/test_1to5_prm_v3_mmiqc.json" \
    --temperature 0.0 \
    --top_k 1 \
    --output_file "/nobackup/users/yikangs/zhiqings/math/outputs/test_1to5_prm_v3_ds-math-7b_mmiqc_v5_1-to-3_g.jsonl" \
    --tensor_parallel_size 1 \
    --resume_generation
