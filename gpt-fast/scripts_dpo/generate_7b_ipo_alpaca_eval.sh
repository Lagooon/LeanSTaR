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
    --finetune_checkpoint_path $DATA_DIR/checkpoints/alpaca-7b-ipo_epoch-2_lr-3e-6_seq-1024_beta-0.001 \
    --batch_size 32 \
    --max_new_tokens 300 \
    --num_samples 1 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/dpo/alpaca_eval_805.json" \
    --temperature 0.7 \
    --top_k 50 \
    --output_file "/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-eval-805_ipo-7b-10k_epoch-2_lr-3e-6_seq-1024_beta-0.001.json" \
    --tensor_parallel_size 1 \
    --resume_generation
