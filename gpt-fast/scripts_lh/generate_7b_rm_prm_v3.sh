set -e
set -x

export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
export ENABLE_INTRA_NODE_COMM=1
module load cuda/11.8
export HF_HOME=/tmp

torchrun --standalone --nproc_per_node=8 \
    batched_rm_score.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_epoch-1_lr-2e-5_seq-768" \
    --batch_size 128 \
    --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_1.jsonl" \
    --output_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_1_score.jsonl" \
    --tensor_parallel_size 1 \
    --on_the_fly_8bit_quantization \
    --resume_generation

torchrun --standalone --nproc_per_node=8 \
    batched_rm_score.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_epoch-1_lr-2e-5_seq-768" \
    --batch_size 128 \
    --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_2.jsonl" \
    --output_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_2_score.jsonl" \
    --tensor_parallel_size 1 \
    --on_the_fly_8bit_quantization \
    --resume_generation

torchrun --standalone --nproc_per_node=8 \
    batched_rm_score.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_epoch-1_lr-2e-5_seq-768" \
    --batch_size 128 \
    --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_3.jsonl" \
    --output_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_3_score.jsonl" \
    --tensor_parallel_size 1 \
    --on_the_fly_8bit_quantization \
    --resume_generation

torchrun --standalone --nproc_per_node=8 \
    batched_rm_score.py \
    --compile \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --finetune_checkpoint_path "$DATA_DIR/checkpoints/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_epoch-1_lr-2e-5_seq-768" \
    --batch_size 128 \
    --prompt_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_4.jsonl" \
    --output_file "/home/wliu/longhui/llms-all/ScalableMath_sun-main/outputs/llemma-7b_metamath_v3_5e-6_1_2_3_tp_288_test_num_samples64_tmp0.7_4_score.jsonl" \
    --tensor_parallel_size 1 \
    --on_the_fly_8bit_quantization \
    --resume_generation
