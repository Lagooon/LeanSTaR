set -e
set -x

export DATA_DIR=/nobackup/users/yikangs/zhiqings/dpo
export MODEL_REPO=huggyllama/llama-7b
export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1

torchrun --standalone --nproc_per_node=8 \
    batched_generate.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/alpaca-7b-10k_epoch-3_lr-1e-4_seq-768" \
    --batch_size 32 \
    --max_new_tokens 300 \
    --num_samples 4 \
    --prompt_file "/nobackup/users/yikangs/zhiqings/dpo/alpaca_unlabeled_noisy_multi_rm_10k.json" \
    --temperature 0.7 \
    --top_k 50 \
    --output_file "/nobackup/users/yikangs/zhiqings/dpo/outputs/alpaca-ualableled-rm-noisy-10k-sft-seq-300-sample-4.json" \
    --tensor_parallel_size 1 \
    --resume_generation
