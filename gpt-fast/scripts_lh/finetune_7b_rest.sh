set -e
set -x
module load cuda/11.8
export DATA_DIR=/lustre/fast/fast/wliu/longhui/trust_math_ckpt
export MODEL_REPO=EleutherAI/llemma_7b
export OMP_NUM_THREADS=8
export ENABLE_INTRA_NODE_COMM=1
export HF_HOME=/tmp

rm -rf $DATA_DIR/checkpoints/llemma-7b_rest_only_2e-5_128_from_sft_tmp0.7
torchrun --standalone --nproc_per_node=8 \
    finetune_rest.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --sft_checkpoint_path $DATA_DIR/checkpoints/llemma-7b_prm_2e-5_128 \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 16 \
    --micro_train_batch_size 8 \
    --learning_rate 5e-6 \
    --lr_eta_min 2e-7 \
    --num_train_epochs 1 \
    --dataset "/home/wliu/longhui/llms-all/ScalableMath_sun-main/data/rest_train_mapped_only_tmp0.7.json" \
    --dataset_format "mapped" \
    --add_eos_to_marked_target \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_dir $DATA_DIR/checkpoints/llemma-7b_rest_only_2e-5_128_from_sft_tmp0.7


bash /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-0106/gpt-fast/scripts_lh/generate_7b_rest_test.sh

python /home/wliu/longhui/llms-all/ScalableMath_sun-main/ScalableMath-0106/gpt-fast/eval_math.py