# set -e
# set -x

# export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
# export MODEL_REPO=EleutherAI/llemma_7b
# export OMP_NUM_THREADS=8
# export ENABLE_INTRA_NODE_COMM=1
# module load cuda/11.8
# export HF_HOME=/tmp

# # /lustre/fast/fast/wliu/longhui/trust_math_ckpt/checkpoints/llemma_7b_1e-5_metamath_mapped/
# torchrun --standalone --nproc_per_node=8 \
#     batched_generate.py \
#     --compile \
#     --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
#     --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_prm_v4_mapped_2e-5" \
#     --batch_size 1 \
#     --max_new_tokens 768 \
#     --num_samples 16 \
#     --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/MATH_test1-3_prompt_processed.json" \
#     --temperature 0.5 \
#     --top_k 100 \
#     --output_file "/lustre/fast/fast/wliu/longhui/workspace/ScalableMath_sun-main/output/test_llemma-7b_prm_v4_mapped_2e-5.jsonl" \
#     --resume_generation


set -e
set -x

export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
export ENABLE_INTRA_NODE_COMM=1
module load cuda/11.8
export HF_HOME=/tmp

rm -rf /home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/test_output.jsonl
torchrun --standalone --nproc_per_node=8 \
    batched_generate.py \
    --default_compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_prm_tp" \
    --batch_size 64 \
    --max_new_tokens 1024 \
    --num_samples 1 \
    --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/prm_splits_MATH_test-cleaned_processed.json" \
    --temperature 0.0 \
    --top_k 20 \
    --output_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/test_output.jsonl" \
    --tensor_parallel_size 8



# conda activate scalable;cd /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-0106/gpt-fast/
# torchrun --standalone --nproc_per_node=8 \
#     batched_generate.py \
#     --compile \
#     --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
#     --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_prm_2e-5_128" \
#     --batch_size 64 \
#     --max_new_tokens 1024 \
#     --num_samples 1 \
#     --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/MATH_test-cleaned_processed.json" \
#     --temperature 0.0 \
#     --top_k 20 \
#     --output_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/output/MATH_test1-5_llemma-7b_prm_2e-5_128_num_samples1.jsonl" \
#     --tensor_parallel_size 8 \
#     --resume_generation