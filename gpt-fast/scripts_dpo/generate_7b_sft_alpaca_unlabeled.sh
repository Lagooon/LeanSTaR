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
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/alpaca-7b-10k_epoch-3_lr-1e-4_seq-768" \
    --batch_size 32 \
    --max_new_tokens 300 \
    --num_samples 2 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/dpo/alpaca_unlabeled_20k.json" \
    --temperature 0.7 \
    --top_k 50 \
    --output_file "/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-unlabeled-20k-sft-seq-300.json" \
    --tensor_parallel_size 1 \
    --resume_generation
