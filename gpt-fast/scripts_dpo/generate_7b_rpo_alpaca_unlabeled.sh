set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/dpo
export MODEL_REPO=huggyllama/llama-7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=4 \
    batched_generate.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/alpaca-7b-rpo_cringe_epoch-2_lr-3e-6_bs-32_beta-0.0002-v2" \
    --batch_size 32 \
    --max_new_tokens 300 \
    --num_samples 4 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/dpo/alpaca_unlabeled_20k.json" \
    --temperature 1.0 \
    --top_k 50 \
    --output_file "/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-unlabeled-20k-rpo-v2-seq-300.json" \
    --tensor_parallel_size 1 \
    --resume_generation
