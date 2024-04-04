set -e
set -x

export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
export ENABLE_INTRA_NODE_COMM=1
module load cuda/11.8
export HF_HOME=/tmp

INDEX=$1

torchrun --standalone --nproc_per_node=2 \
    batched_generate.py \
    --default_compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288" \
    --batch_size 64 \
    --max_new_tokens 1024 \
    --num_samples 64 \
    --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/prm_splits_MATH_test-cleaned_processed.json" \
    --temperature 0.7 \
    --top_k 20 \
    --output_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_${INDEX}.jsonl" \
    --tensor_parallel_size 2
# bash /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-0118/gpt-fast/scripts_lh/generate_7b_sft_test_maj.sh 1
# conda activate scalable;cd /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-0118/gpt-fast
# bash /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-0106/gpt-fast/scripts_lh/generate_7b_sft_rest.sh
# conda activate scalable;cd /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-0106/gpt-fast/