set -e
set -x

export DATA_DIR=/workspace/zhiqings/output4/math
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
# export NCCL_IB_TIMEOUT=22
# export ENABLE_INTRA_NODE_COMM=1
# export NCCL_CROSS_NIC=1
# export CUDA_LAUNCH_BLOCKING=1

ls data_utils
ls grading
ls models
ls training_utils

torchrun --standalone --nproc_per_node=8 \
    batched_generate.py \
    --default_compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b-ppo_metamath-1to5_epoch-50_lr-1e-6_seq-768-vd" \
    --batch_size 32 \
    --max_new_tokens 1024 \
    --num_samples 1 \
    --prompt_file "$DATA_DIR/test_1to5_prm_v3_max256.json" \
    --temperature 0.0 \
    --top_k 1 \
    --output_file "$DATA_DIR/outputs/test_1to5_prm_v3_max256_7b-metamath_ppo-vd_g.jsonl" \
    --tensor_parallel_size 8 \
    --finetune_checkpoint_prefix "policy_" \
    --resume_generation
